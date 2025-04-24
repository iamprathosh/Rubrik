import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsgluedq.transforms import EvaluateDataQuality
from awsglue import DynamicFrame

def query_dynamic_frames(glue_context, sql_query, view_map, context_id):
    """
    Run SQL queries on dynamic frames by creating temporary views.
    
    Args:
        glue_context: The GlueContext to use
        sql_query: SQL query string to execute
        view_map: Dictionary mapping view names to DynamicFrames
        context_id: Transformation context identifier
        
    Returns:
        A new DynamicFrame with the query results
    """
    # Create temporary views for each dynamic frame
    for view_name, frame in view_map.items():
        frame.toDF().createOrReplaceTempView(view_name)
        
    # Execute SQL query
    query_output = spark.sql(sql_query)
    
    # Convert back to DynamicFrame
    return DynamicFrame.fromDF(query_output, glue_context, context_id)

# Initialize the Glue job environment
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Define data quality check rules
DATA_QUALITY_RULES = """
    Rules = [
        ColumnCount > 0
    ]
"""

# Load step trainer readings from landing zone
step_trainer_landing = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://cd0030bucket/step_trainer/"]},
    format="json",
    transformation_ctx="step_trainer_landing_source"
)

# Load curated customer data (customers who have consented and have accelerometer data)
customer_curated = glueContext.create_dynamic_frame.from_catalog(
    database="stedi_db", 
    table_name="customer_curated", 
    transformation_ctx="customer_curated_source"
)

# Filter step trainer readings to include only those from customers in the curated zone
# This ensures we only process data for customers who have consented to research
step_trainer_filter_query = """
SELECT trainer_data.* 
FROM trainer_data
INNER JOIN customer_data
ON trainer_data.serialnumber = customer_data.serialnumber
"""

step_trainer_trusted = query_dynamic_frames(
    glueContext,
    step_trainer_filter_query,
    {"trainer_data": step_trainer_landing, "customer_data": customer_curated},
    "filter_step_trainer_to_trusted"
)

# Validate data quality before saving
EvaluateDataQuality().process_rows(
    frame=step_trainer_trusted, 
    ruleset=DATA_QUALITY_RULES, 
    publishing_options={
        "dataQualityEvaluationContext": "step_trainer_trusted_quality_check", 
        "enableDataQualityResultsPublishing": True
    }, 
    additional_options={
        "dataQualityResultsPublishing.strategy": "BEST_EFFORT", 
        "observations.scope": "ALL"
    }
)

# Write trusted step trainer data to S3
trusted_step_trainer_sink = glueContext.getSink(
    path="s3://stedi-human-sr/step_trainer/trusted/", 
    connection_type="s3", 
    updateBehavior="UPDATE_IN_DATABASE", 
    partitionKeys=[], 
    enableUpdateCatalog=True, 
    transformation_ctx="write_step_trainer_trusted"
)

trusted_step_trainer_sink.setCatalogInfo(
    catalogDatabase="stedi_db",
    catalogTableName="step_trainer_trusted"
)
trusted_step_trainer_sink.setFormat("json")
trusted_step_trainer_sink.writeFrame(step_trainer_trusted)

job.commit()
