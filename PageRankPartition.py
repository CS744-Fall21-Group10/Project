"""PageRank.py"""
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import lit, sum, coalesce, col, lower
import sys

if __name__ == "__main__":
    #For accessing files on HDFS
    hdfsPrefix="hdfs://10.10.1.1:9000/"
    # Input file is first argument
    in_file = hdfsPrefix+sys.argv[1]
    # Output directory is second argument
    out_file = hdfsPrefix+sys.argv[2]

    #set this to control number of partitions
    numParts=67

    # Load spark session and run locally
    spark = SparkSession \
        .builder \
        .appName("PageRank") \
        .config("spark.driver.memory", "30G") \
        .config("spark.executor.memory", "30G") \
        .config("spark.driver.cores",1) \
        .config("spark.task.cpus",1) \
        .master("spark://c220g1-030827vm-1.wisc.cloudlab.us:7077") \
        .getOrCreate()

    # Schema for edge list pairs
    adj_schema = StructType([
        StructField("from", StringType(), False),
        StructField("to", StringType(), False)
        ])
    # Read tab-separated edge list
    adj = spark.read.option("comment", "#") \
        .csv(in_file, sep='\t', schema=adj_schema)

    #lower casing
    adj = adj.withColumn("from", lower(col("from"))) \
       .withColumn("to", lower(col("to")))

    # Get outdegrees
    outdeg = adj.groupBy("from").count() \
        .withColumnRenamed("count", "outdeg")

    # Add outdegrees to edge list
    adj = adj.join(outdeg, on="from")

    #repartition differently
    adj = adj.repartition(numParts)

    # Starting ranks for intial vertices
    rank = adj.select("from").distinct() \
        .withColumn("rank", lit(1.0))
    default_rank = adj.select("from").distinct() \
        .withColumn("default_rank", lit(0.15))
    

    N = 10

    for i in range(N):
        # Find rank of each initial vertex
        df = adj.join(rank, on="from")
        # Divide by outdegree
        df = df.withColumn("contrib", df["rank"] / df["outdeg"])
        # Sum all contributions to final vertex
        rank = df.groupBy("to").agg(sum("contrib").alias("contribs"))
        # Include 15% random restart
        rank = rank.withColumn("rank", 0.15 + 0.85 * rank["contribs"])
        # Rename final vertex to initial vertex as sum was over final
        rank = rank.withColumnRenamed("to", "from")
        #Add proper ranks for in-degree 0 nodes
        rank = rank.join(default_rank, on="from", how="outer")
        rank = rank.withColumn("rank", coalesce("rank", "default_rank"))

    # Format final output as (node,rank) pairs
    rank = rank.select("from", "rank") \
        .withColumnRenamed("from", "node") \
        .write.csv(out_file, header="True")

