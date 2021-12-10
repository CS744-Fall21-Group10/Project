"""PageRank.py"""
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.window import Window
from pyspark.sql.functions import lit, sum, max, coalesce, rand, desc, when, greatest, first, col, row_number
from pyspark.sql.functions import row_number, monotonically_increasing_id, UserDefinedFunction
from pyspark.sql import Window
from pyspark.sql.types import IntegerType
import sys
from datetime import datetime

def getTimeFromStart(startTime):
    return (datetime.now() - startTime).total_seconds()

def dp_partition(adj, partition, T)
    part_from = partition.withColumnRenamed("partition", "part_from")

    # Forward DP steps
    for t in range(T):
        part_to = part_from \
            .withColumnRenamed("from", "to")
            .withColumnRenamed("part_from", "part_to")

        # Add partitions of nodes to edges
        part_adj = adj \
            .merge(part_from, on="from") \
            .marge(part_to, on="to")

        # Find intra- and inter-partition edges
        intra_edges = part_adj.filter(part_adj["part_from"] == part_adj["part_to"])
        inter_edges = part_adj.filter(part_adj["part_from"] != part_adj["part_to"])

        # Compute size of partitions (number of intra-partition edges)
        intra_size = (intra_edges.groupBy("part_from").count().withColumnRenamed("part_from", "part") \
                + intra_edges.groupBy("part_to").count().withColumnRenamed("part_to", "part")) \
            .withColumnRenamed("count", "part_size")
        max_sizes = intra_size.orderBy("part_size", ascending=False).take(2).collect()
        size1 = max_sizes[0]["part_size"]
        size2 = max_sizes[1]["part_size"]

        # Compute size of cuts (number of inter-partition edges)
        inter_from = inter_edges.groupBy("part_from").count()
        inter_to = inter_edges.groupBy("part_to").count()

        # Number of edges to each node from each partition
        num_to = part_adj.groupBy("to", "part_from")
        num_to = num_to.count() \
            .withColumnRenamed("to", "node") \
            .withColumnRenamed("part_from", "new_part") \
            .withColumnRenamed("part_to", "old_part")
        num_to_diff = num_to.filter(num_to["new_part"] != num_to["old_part"]) \
            .withColumnRenamed("count", "num_to_new")
        num_to_same = num_to.filter(num_to["new_part"] == num_to["old_part"]) \
            .withColumnRenamed("count", "num_to_old")

        # Number of edges from each node to each partition
        num_from = part_adj.groupBy("from", "part_to")
        num_from = num_from.count() \
            .withColumnRenamed("from", "node") \
            .withColumnRenamed("part_to", "new_part") \
            .withColumnRenamed("part_from", "old_part")
        num_from_diff = num_from.filter(num_from["new_part"] != num_from["old_part"]) \
            .withColumnRenamed("count", "num_from_new")
        num_from_same = num_from.filter(num_from["new_part"] == num_from["old_part"]) \
            .withColumnRenamed("count", "num_from_old")

        # Compute costs for each possible new partition
        options = num_to_diff.join(num_from_diff, on=["node", "old_part", "new_part"], how="outer").fillna(0) \
            .merge(num_to_same, on="node") \
            .merge(num_from_same, on="node")
        options["inter_diff"] = options["num_to_old"] + options["num_from_old"] - options["num_to_new"] - options["num_from_new"]
        intra_new = intra_size.withColumnRenamed("part", "new_part") \
            .withColumnRenamed("part_size", "new_part_size")
        options = options.merge(intra_new, on=["new_part"])
        options["new_part_size"] = options["new_part_size"] + options["num_to_old"] + options["num_from_old"]
        options["second_part_size"] = lit(size2)
        options["old_max_part_size"] = lit(size1)
        options["new_max_part_size"] = lit(size1) - options["num_to_old"] - options["num_from_old"]
        options["new_intra_max"] =
            when(options["part_size"] == size1,
                greatest("new_part_size", "second_part_size", "new_max_part_size")) \
            .otherwise(
                greatest("new_part_size", "old_max_part_size")
                )
        options["intra_diff"] = options["new_intra_max"] - options["old_max_part_size"]
        options["cost_diff"] = options["inter_diff"] + options["intra_diff"]

        # Take best option for each node independently
        lowest_cost = Window.partitionBy("node").orderBy(col("cost_diff").asc())
        best_option = options.withColumn("row_number", row_number().over(lowest_cost)) \
            .filter(col("row_number") == 1).drop("row")

        part_from = best_option.select("node", "new_part") \
            .withColumnRenamed("node", "from") \
            .withColumnRenamed("new_part", "part_from")

    return part_from.withColumnRenamed("part_from", "partition")

if __name__ == "__main__":
    #For accessing files on HDFS
    hdfsPrefix = "hdfs://10.10.1.1:9000/"
    # Input file is first argument
    in_file = hdfsPrefix + sys.argv[1]
    # Output directory is second argument
    out_file = hdfsPrefix + sys.argv[2]

    # Load spark session and run locally
    spark = SparkSession \
        .builder \
        .appName("PageRank") \
        .config("spark.driver.memory", "12G") \
        .config("spark.executor.memory", "12G") \
        .config("spark.driver.cores",1) \
        .config("spark.task.cpus",1) \
        .master("spark://c220g5-111226vm-1.wisc.cloudlab.us:7077") \
        .getOrCreate()

    # Schema for edge list pairs
    adj_schema = StructType([
        StructField("from", IntegerType(), False),
        StructField("to", IntegerType(), False)
        ])
    # Read tab-separated edge list
    adj = spark.read.option("comment", "#") \
        .csv(in_file, sep='\t', schema=adj_schema)

    # Get degrees
    outdeg = adj.groupBy("from").count() \
        .withColumnRenamed("count", "outdeg")
    indeg = adj.groupBy("to").count() \
        .withColumnRenamed("count", "indeg")

    # Add outdegrees to edge list
    adj = adj.join(outdeg, on="from")

    # Starting ranks for intial vertices
    rank = adj.select("from").distinct() \
        .withColumn("rank", lit(1.0))
    default_rank = adj.select("from").distinct() \
        .withColumn("default_rank", lit(0.15))

    adj = adj.withColumn("index",
        row_number().over(Window.orderBy(monotonically_increasing_id()))-1
    )

    # Number of partitions
    k = 200
    # Number of steps
    N = 10
    # Number of DP iterations per step
    T = 2

    print("Beginning partitioning...")
    startTime = datetime.now()
    partition = adj.select("from").distinct() \
        .withColumn("partition", (rand()*k).cast("int"))
    adj["partition"] = partition
    adj = adj.repartition(k, "partition")

    partition = dp_partition(adj, part_from, T)
    adj["partition"] = partition
    adj = adj.repartition(numParts, "partition")
    adj = adj.repartition(k, "partition")
    
    for i in range(N):
        if i == 0:
            print("Completed partitioning, time taken:",
                    getTimeFromStart(startTime))
            startTime = datetime.now()
        else:
            print("PageRank iter complete: ",
                    getTimeFromStart(startTime))
            startTime = datetime.now()

        # Find rank of each initial vertex
        df = adj.join(rank, on="from")
        # Divide by outdegree
        df = df.withColumn("contrib", df["rank"] / df["outdeg"])
        # Sum all contributions to final vertex
        rank = df.groupBy("to").agg(sum("contrib").alias("contribs"))
        # Include 15% random restart
        rank = rank.withColumn("rank1", 0.15 + 0.85 * rank["contribs"])
        # Rename final vertex to initial vertex as sum was over final
        rank = rank.withColumnRenamed("to", "from")
        # Add proper ranks for in-degree 0 nodes
        rank = rank.join(default_rank, on="from", how="outer")
        rank = rank.withColumn("rank", coalesce("rank1", "default_rank"))

        #partition = dp_partition(adj, part_from, T)
        #adj["partition"] = partition
        #adj = adj.repartition(numParts, "partition")
        #adj = adj.repartition(k, "partition")
        
    # Format final output as (node,rank) pairs
    rank = rank.select("from", "rank") \
        .withColumnRenamed("from", "node") \
        .coalesce(1) \
        .write.csv(out_file, header="True")
