import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsgluedq.transforms import EvaluateDataQuality
from awsglue import DynamicFrame

def process_sql_query(glue_ctx, sql, dataframes, ctx_name):
    """Execute SQL query on dynamic frames and return result as a DynamicFrame."""
    for view_name, dframe in dataframes.items():
        dframe.toDF().createOrReplaceTempView(view_name)
    result_df = spark.sql(sql)
    return DynamicFrame.fromDF(result_df, glue_ctx, ctx_name)

# Set up the Glue job
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Data quality rule definition
DATA_QUALITY_RULES = """
    Rules = [
        ColumnCount > 0
    ]
"""

# Extract customer data from landing zone
customer_landing = glueContext.create_dynamic_frame.from_catalog(
    database="stedi_db", 
    table_name="customer_landing", 
    transformation_ctx="customer_landing_extraction"
)

# Filter customers who have agreed to share their data for research
customer_query = '''
SELECT * FROM customer_source
WHERE sharewithresearchasofdate IS NOT NULL
'''

customer_trusted = process_sql_query(
    glueContext,
    customer_query,
    {"customer_source": customer_landing},
    "customer_trusted_transformation"
)

# Evaluate data quality before saving
EvaluateDataQuality().process_rows(
    frame=customer_trusted,
    ruleset=DATA_QUALITY_RULES,
    publishing_options={
        "dataQualityEvaluationContext": "customer_trusted_quality_check",
        "enableDataQualityResultsPublishing": True
    },
    additional_options={
        "dataQualityResultsPublishing.strategy": "BEST_EFFORT",
        "observations.scope": "ALL"
    }
)

# Save filtered data to trusted zone
trusted_output = glueContext.getSink(
    path="s3://stedi-human-sr/customer/trusted/",
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=[],
    enableUpdateCatalog=True,
    transformation_ctx="customer_trusted_storage"
)

trusted_output.setCatalogInfo(catalogDatabase="stedi_db", catalogTableName="customer_trusted")
trusted_output.setFormat("json")
trusted_output.writeFrame(customer_trusted)

job.commit()