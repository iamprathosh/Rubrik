import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsgluedq.transforms import EvaluateDataQuality
from awsglue import DynamicFrame

def execute_sql(context, query_text, view_mappings, transformation_name):
    """
    Utility function to run SQL queries against dynamic frames
    
    Parameters:
        context: GlueContext instance
        query_text: SQL query to execute
        view_mappings: Dictionary mapping view names to DynamicFrames
        transformation_name: Name for the transformation context
        
    Returns:
        DynamicFrame with query results
    """
    # Register each frame as a temporary view
    for view_name, dframe in view_mappings.items():
        dframe.toDF().createOrReplaceTempView(view_name)
    
    # Execute the query
    sql_output = spark.sql(query_text)
    
    # Convert back to DynamicFrame
    return DynamicFrame.fromDF(sql_output, context, transformation_name)

# Initialize Glue job
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Define data quality rules
QUALITY_VALIDATION_RULES = """
    Rules = [
        ColumnCount > 0
    ]
"""

# Read accelerometer data from trusted zone (only from opted-in customers)
accelerometer_trusted = glueContext.create_dynamic_frame.from_catalog(
    database="stedi_db", 
    table_name="accelerometer_trusted", 
    transformation_ctx="load_accelerometer_trusted"
)

# Read customer data from trusted zone (only customers who opted in)
customer_trusted = glueContext.create_dynamic_frame.from_catalog(
    database="stedi_db", 
    table_name="customer_trusted", 
    transformation_ctx="load_customer_trusted"
)

# Join customer data with accelerometer readings to create curated dataset
# This ensures we only include customers who have accelerometer data
customer_with_accelerometer = Join.apply(
    frame1=accelerometer_trusted, 
    frame2=customer_trusted, 
    keys1=["user"], 
    keys2=["email"], 
    transformation_ctx="join_customer_accelerometer"
)

# Extract unique customer records with all relevant fields
curated_customer_query = '''
SELECT DISTINCT 
    customername, 
    email, 
    phone, 
    birthday,
    serialnumber, 
    registrationdate, 
    lastupdatedate, 
    sharewithresearchasofdate,
    sharewithpublicasofdate, 
    sharewithfriendsasofdate
FROM customer_accelerometer_data
'''

customer_curated = execute_sql(
    glueContext,
    curated_customer_query,
    {"customer_accelerometer_data": customer_with_accelerometer},
    "extract_curated_customers"
)

# Verify data quality
EvaluateDataQuality().process_rows(
    frame=customer_curated, 
    ruleset=QUALITY_VALIDATION_RULES, 
    publishing_options={
        "dataQualityEvaluationContext": "customer_curated_quality_validation", 
        "enableDataQualityResultsPublishing": True
    }, 
    additional_options={
        "dataQualityResultsPublishing.strategy": "BEST_EFFORT", 
        "observations.scope": "ALL"
    }
)

# Write curated customer data to S3 and update Data Catalog
curated_customer_sink = glueContext.getSink(
    path="s3://stedi-human-sr/customer/curated/", 
    connection_type="s3", 
    updateBehavior="UPDATE_IN_DATABASE", 
    partitionKeys=[], 
    enableUpdateCatalog=True, 
    transformation_ctx="write_customer_curated"
)

curated_customer_sink.setCatalogInfo(
    catalogDatabase="stedi_db",
    catalogTableName="customer_curated"
)
curated_customer_sink.setFormat("json")
curated_customer_sink.writeFrame(customer_curated)

job.commit()