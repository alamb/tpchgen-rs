//! TPC-DS Parquet output.

use super::generate::output_location_for_table;
use super::runner::PlannedTable;
use crate::output_location::OutputLocation;
use crate::parquet::ParquetOutput;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatchReader;
use log::info;
use parquet::basic::{Compression, Encoding};
use std::io;
use tpcdsgen::config::{Session, Table};
use tpcdsgen_arrow::{
    CallCenterArrow, CatalogPageArrow, CatalogReturnsArrow, CatalogSalesArrow,
    CustomerAddressArrow, CustomerArrow, CustomerDemographicsArrow, DateDimArrow,
    DbgenVersionArrow, HouseholdDemographicsArrow, IncomeBandArrow, InventoryArrow, ItemArrow,
    PromotionArrow, ReasonArrow, ShipModeArrow, StoreArrow, StoreReturnsArrow, StoreSalesArrow,
    TimeDimArrow, WarehouseArrow, WebPageArrow, WebReturnsArrow, WebSalesArrow, WebSiteArrow,
};

/// Parquet files can have at most 32767 row groups
pub(super) const MAX_ROW_GROUPS: u64 = 32767;

fn table_schema(table: Table) -> SchemaRef {
    match table {
        Table::CallCenter => CallCenterArrow::schema_ref(),
        Table::CatalogPage => CatalogPageArrow::schema_ref(),
        Table::CatalogReturns => CatalogReturnsArrow::schema_ref(),
        Table::CatalogSales => CatalogSalesArrow::schema_ref(),
        Table::Customer => CustomerArrow::schema_ref(),
        Table::CustomerAddress => CustomerAddressArrow::schema_ref(),
        Table::CustomerDemographics => CustomerDemographicsArrow::schema_ref(),
        Table::DateDim => DateDimArrow::schema_ref(),
        Table::DbgenVersion => DbgenVersionArrow::schema_ref(),
        Table::HouseholdDemographics => HouseholdDemographicsArrow::schema_ref(),
        Table::IncomeBand => IncomeBandArrow::schema_ref(),
        Table::Inventory => InventoryArrow::schema_ref(),
        Table::Item => ItemArrow::schema_ref(),
        Table::Promotion => PromotionArrow::schema_ref(),
        Table::Reason => ReasonArrow::schema_ref(),
        Table::ShipMode => ShipModeArrow::schema_ref(),
        Table::Store => StoreArrow::schema_ref(),
        Table::StoreReturns => StoreReturnsArrow::schema_ref(),
        Table::StoreSales => StoreSalesArrow::schema_ref(),
        Table::TimeDim => TimeDimArrow::schema_ref(),
        Table::Warehouse => WarehouseArrow::schema_ref(),
        Table::WebPage => WebPageArrow::schema_ref(),
        Table::WebReturns => WebReturnsArrow::schema_ref(),
        Table::WebSales => WebSalesArrow::schema_ref(),
        Table::WebSite => WebSiteArrow::schema_ref(),
        _ => unreachable!("table_schema is only called for main TPC-DS tables"),
    }
}

/// Checks each column in `encodings` against every table in `tables`.
///
/// Rejects an encoding `reject_unsupported_encoding` always rejects.
/// Rejects a column name that matches no table (almost always a typo). A
/// column that matches only some tables is fine: [`column_encodings_for_table`]
/// applies it there and skips it elsewhere.
fn validate_column_encodings(tables: &[Table], encodings: &[(String, Encoding)]) -> io::Result<()> {
    for (col, enc) in encodings {
        crate::parquet::reject_unsupported_encoding(*enc)?;
        let matches_any_table = tables.iter().any(|table| {
            table_schema(*table)
                .fields()
                .iter()
                .any(|f| f.name() == col)
        });
        if !matches_any_table {
            return Err(io::Error::other(format!(
                "column '{col}' for --column-encoding not found in any selected table"
            )));
        }
    }
    Ok(())
}

/// Keeps only the encodings whose column exists in `table`'s schema.
fn column_encodings_for_table(
    table: Table,
    encodings: &[(String, Encoding)],
) -> Vec<(String, Encoding)> {
    let schema = table_schema(table);
    encodings
        .iter()
        .filter(|(col, _)| schema.fields().iter().any(|f| f.name() == col))
        .cloned()
        .collect()
}

/// Parquet output generator.
#[derive(Debug, Clone)]
pub(super) struct Parquet {
    base_location: OutputLocation,
    pub(super) compression: Compression,
    pub(super) row_group_bytes: i64,
    column_encodings: Option<Vec<(String, Encoding)>>,
    field_ids: bool,
}

impl Parquet {
    pub(super) fn new(
        base_location: OutputLocation,
        compression: Compression,
        row_group_bytes: i64,
        column_encodings: Option<Vec<(String, Encoding)>>,
        field_ids: bool,
    ) -> Self {
        Self {
            base_location,
            compression,
            row_group_bytes,
            column_encodings,
            field_ids,
        }
    }

    /// Reject a `--column-encoding` column that matches no selected table
    /// (a typo) before any work starts. `column_encodings_for_table` skips a
    /// column that only matches some tables, so that case is not an error.
    pub(super) fn validate(&self, table_sessions: &[(Table, Session)]) -> io::Result<()> {
        if let Some(encodings) = &self.column_encodings {
            let selected_tables: Vec<Table> =
                table_sessions.iter().map(|(table, _)| *table).collect();
            validate_column_encodings(&selected_tables, encodings)?;
        }
        Ok(())
    }

    /// Write one table to a Parquet file at the specified path.
    ///
    /// `make_reader` creates a [`RecordBatchReader`] for one planned source
    /// row range; the batches of each reader are encoded (in parallel, using
    /// up to `num_threads` threads) as one row group.
    ///
    /// Progress is reported in row groups: the shared writer advances by
    /// one per written row group (the same output units as TPC-H parquet
    /// generation; the totals are registered by [`super::runner::plan_tables`]).
    pub(super) async fn write_table<R, F>(
        &self,
        planned: PlannedTable,
        num_threads: usize,
        make_reader: F,
    ) -> io::Result<()>
    where
        R: RecordBatchReader + Send + 'static,
        F: Fn(Session, u64, u64) -> R + Send + 'static,
    {
        let PlannedTable {
            table,
            session,
            plan,
            progress,
        } = planned;

        // Keep only the encodings for columns on this table.
        // --column-encoding usually targets a few tables, not all of them.
        let column_encodings = self
            .column_encodings
            .as_ref()
            .map(|encodings| column_encodings_for_table(table, encodings));

        let location = output_location_for_table(&self.base_location, table, "parquet", &session)?;
        let chunk_count = plan.chunk_count() as u64;
        let scale_factor = session.get_scaling().get_scale();
        let part = session.get_chunk_number();
        let parts = session.get_total_chunks();
        let partition = if session.is_partitioned() {
            format!(" (part {part}/{parts})")
        } else {
            String::new()
        };
        let sources = plan
            .into_iter()
            .map(move |range| make_reader(session.clone(), *range.start(), *range.end()));

        info!(
            "Writing table {table} (SF={scale_factor}, {chunk_count} chunk{}){partition} to {location} using {num_threads} thread{}",
            if chunk_count == 1 { "" } else { "s" },
            if num_threads == 1 { "" } else { "s" }
        );
        let written = location
            .write(ParquetOutput {
                sources,
                num_threads,
                compression: self.compression,
                column_encodings: column_encodings.as_deref(),
                field_ids: self.field_ids,
                progress: progress.clone(),
            })
            .await?;
        if written {
            info!("Generated table {table}{partition} to {location}");
        } else {
            // Skipped, so count all chunks at once
            progress.increment(chunk_count, 0);
        }
        progress.complete();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_column_encodings_accepts_a_column_present_on_just_one_table() {
        // r_reason_desc exists only on reason, not item.
        let tables = [Table::Reason, Table::Item];
        let encodings = [("r_reason_desc".to_string(), Encoding::PLAIN)];
        assert!(validate_column_encodings(&tables, &encodings).is_ok());
    }

    #[test]
    fn validate_column_encodings_rejects_a_typo() {
        let tables = [Table::Reason];
        let encodings = [("r_reason_desc_typo".to_string(), Encoding::PLAIN)];
        let err = validate_column_encodings(&tables, &encodings).unwrap_err();
        assert!(
            err.to_string().contains("column 'r_reason_desc_typo'"),
            "{err}"
        );
    }

    #[test]
    fn validate_column_encodings_rejects_dictionary_encoding() {
        // The column is real, so the only reason to fail is the encoding.
        let tables = [Table::Reason];
        let encodings = [("r_reason_desc".to_string(), Encoding::PLAIN_DICTIONARY)];
        let err = validate_column_encodings(&tables, &encodings).unwrap_err();
        assert!(err.to_string().contains("dictionary encoding"), "{err}");
    }

    #[test]
    fn column_encodings_for_table_keeps_only_matching_columns() {
        let encodings = [
            ("r_reason_desc".to_string(), Encoding::PLAIN),
            ("i_item_desc".to_string(), Encoding::PLAIN),
        ];
        assert_eq!(
            column_encodings_for_table(Table::Reason, &encodings),
            vec![("r_reason_desc".to_string(), Encoding::PLAIN)]
        );
        assert_eq!(
            column_encodings_for_table(Table::CallCenter, &encodings),
            Vec::new()
        );
    }
}
