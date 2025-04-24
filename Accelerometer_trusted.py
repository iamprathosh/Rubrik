import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsgluedq.transforms import EvaluateDataQuality
from awsglue import DynamicFrame

def run_sql_transformation(glue_ctx, sql_statement, frame_dict, context_name):
    """
    Execute a SQL query against temporary views created from dynamic frames
    
    Args:
        glue_ctx: GlueContext to use
        sql_statement: SQL query to execute
        frame_dict: Dict mapping view names to DynamicFrames
        context_name: Transformation context name
        
    Returns:
        DynamicFrame with query results
    """
    for view_name, dynamic_frame in frame_dict.items():
        dynamic_frame.toDF().createOrReplaceTempView(view_name)
    
    sql_result = spark.sql(sql_statement)
    return DynamicFrame.fromDF(sql_result, glue_ctx, context_name)

# Initialize the Glue job
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Define data quality validation rules
QUALITY_RULESET = """
    Rules = [
        ColumnCount > 0
    ]
"""

# Load raw accelerometer data from landing zone
accelerometer_landing = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://cd0030bucket/accelerometer/"]},
    format="json",
    transformation_ctx="accelerometer_landing_source"
)

# Load validated customer data from trusted zone
customer_trusted = glueContext.create_dynamic_frame.from_catalog(
    database="stedi_db", 
    table_name="customer_trusted", 
    transformation_ctx="customer_trusted_source"
)

# Join accelerometer data with trusted customers to filter for consenting users only
accelerometer_customer_joined = Join.apply(
    frame1=accelerometer_landing,
    frame2=customer_trusted,
    keys1=["user"],
    keys2=["email"],
    transformation_ctx="accelerometer_customer_join"
)

# Extract only necessary accelerometer fields
extraction_query = '''
SELECT user, timestamp, x, y, z
FROM accelerometer_data
'''

accelerometer_trusted = run_sql_transformation(
    glueContext,
    extraction_query,
    {"accelerometer_data": accelerometer_customer_joined},
    "accelerometer_extraction"
)

# Verify data quality before saving
EvaluateDataQuality().process_rows(
    frame=accelerometer_trusted, 
    ruleset=QUALITY_RULESET, 
    publishing_options={
        "dataQualityEvaluationContext": "accelerometer_trusted_quality_check",
        "enableDataQualityResultsPublishing": True
    }, 
    additional_options={
        "dataQualityResultsPublishing.strategy": "BEST_EFFORT", 
        "observations.scope": "ALL"
    }
)

# Write filtered accelerometer data to trusted zone
trusted_accelerometer_sink = glueContext.getSink(
    path="s3://stedi-human-sr/accelerometer/trusted/", 
    connection_type="s3", 
    updateBehavior="UPDATE_IN_DATABASE", 
    partitionKeys=[], 
    enableUpdateCatalog=True, 
    transformation_ctx="accelerometer_trusted_sink"
)

trusted_accelerometer_sink.setCatalogInfo(
    catalogDatabase="stedi_db",
    catalogTableName="accelerometer_trusted"
)
trusted_accelerometer_sink.setFormat("json")
trusted_accelerometer_sink.writeFrame(accelerometer_trusted)

job.commit()
