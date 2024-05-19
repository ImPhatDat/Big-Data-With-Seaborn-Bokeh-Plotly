import findspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f
import argparse
import os

step_files = 100
goldman = [
    [-74.0141012, 40.7152191],
    [-74.013777, 40.7152275],
    [-74.0141027, 40.7138745],
    [-74.0144185, 40.7140753],
]
citigroup = [
    [-74.011869, 40.7217236],
    [-74.009867, 40.721493],
    [-74.010140, 40.720053],
    [-74.012083, 40.720267],
]
yellow_schema = StructType(
    [
        StructField("Type", StringType(), nullable=False),
        StructField("VendorID", IntegerType()),
        StructField("Pickup_datetime", TimestampType()),
        StructField("Dropoff_datetime", TimestampType()),
        StructField("Passenger_count", IntegerType()),
        StructField("Trip_distance", FloatType()),
        StructField("Pickup_longitude", FloatType()),
        StructField("Pickup_latitude", FloatType()),
        StructField("Rate_codeID", FloatType()),
        StructField("Store_and_fwd_flag", StringType()),
        StructField("Dropoff_longitude", FloatType()),
        StructField("Dropoff_latitude", FloatType()),
        StructField("Payment_type", IntegerType()),
        StructField("Fare_amount", FloatType()),
        StructField("Extra", FloatType()),
        StructField("Mta_tax", FloatType()),
        StructField("Tip_amount", FloatType()),
        StructField("Tolls_amount", FloatType()),
        StructField("Improvement_surcharge", FloatType()),
        StructField("Total_amount", FloatType()),
    ]
)
green_schema = StructType(
    [
        StructField("Type", StringType(), nullable=False),
        StructField("VendorID", IntegerType()),
        StructField("Pickup_datetime", TimestampType()),
        StructField("Dropoff_datetime", TimestampType()),
        StructField("Store_and_fwd_flag", StringType()),
        StructField("Rate_codeID", IntegerType()),
        StructField("Pickup_longitude", FloatType()),
        StructField("Pickup_latitude", FloatType()),
        StructField("Dropoff_longitude", FloatType()),
        StructField("Dropoff_latitude", FloatType()),
        StructField("Passenger_count", IntegerType()),
        StructField("Trip_distance", FloatType()),
        StructField("Fare_amount", FloatType()),
        StructField("Extra", FloatType()),
        StructField("Mta_tax", FloatType()),
        StructField("Tip_amount", FloatType()),
        StructField("Tolls_amount", FloatType()),
        StructField("Ehail_fee", FloatType()),
        StructField("Improvement_surcharge", FloatType()),
        StructField("Total_amount", FloatType()),
        StructField("Payment_type", IntegerType()),
        StructField("Trip_type", IntegerType()),
    ]
)
full_schema = StructType(
    [StructField(f"_c{i}", StringType(), nullable=True) for i in range(22)]
)


@f.udf(returnType=BooleanType())
def in_polygon(x, y, polygon):
    num_vertices = len(polygon)
    inside = False
    
    # first point
    p1 = polygon[0]
    
    # For each edge
    for i in range(1, num_vertices + 1):
        # Next point
        p2 = polygon[i % num_vertices]
        
        # if is above the minimum latitude (y)
        if y > min(p1[1], p2[1]):
            # if is below the maximum latitude (y)
            if y <= max(p1[1], p2[1]):
                # if is to the left of the maximum longitude (x)
                if x <= max(p1[0], p2[0]):
                    # get intersection between horizontal line and the edge
                    x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    # if the point is on the same line as the edge or to the left of the x-intersection
                    if p1[0] == p2[0] or x <= x_intersection:
                        # modify flag
                        inside = not inside
                        
        # Store the current point as the first point for the next iteration
        p1 = p2
    # Return final value of the flag
    return inside


def foreach_batch_function(output_path):
    def batch_func(batch_df, batch_id):
        # Create data: A
        windowed_df = batch_df.withColumn(
            "window_start", f.col("window.start").cast("long")
        )
        windowed_df = windowed_df.withColumn(
            "prev_window_start", f.col("window_start") - 600
        )

        # Get data and format it to prev_window_start : B
        previous_df = windowed_df.select(
            f.col("headquarter").alias("prev_headquarter"),
            f.col("count").alias("prev_count"),
            f.col("window_start").alias("prev_window_start"),
        )

        # A and B: (window.start - headquarter - count - window_start - prev_window_start - prev_count)
        joined_df = windowed_df.join(
            previous_df,
            (windowed_df.headquarter == previous_df.prev_headquarter)
            & (windowed_df.prev_window_start == previous_df.prev_window_start),
            "left_outer",
        ).fillna(0, subset=["prev_count"])

        joined_df = joined_df.withColumn("start_hour", f.hour(f.col("window.start")))
        joined_df = joined_df.withColumn("start_min", f.minute(f.col("window.start")))

        # Column bool check trend
        def detect_trend(current_count, previous_count, hour, minute, headquarter):
            if current_count >= 10 and current_count >= 2 * previous_count:
                return f"({headquarter},({current_count},{(hour * 60 + minute) * 60 * 1000},{previous_count}))"
            else:
                return None

        detect_trend_udf = f.udf(detect_trend, StringType())

        final_df = joined_df.withColumn(
            "trend",
            detect_trend_udf(
                f.col("count"),
                f.col("prev_count"),
                f.col("start_hour"),
                f.col("start_min"),
                f.col("headquarter"),
            ),
        ).filter(f.col("trend").isNotNull())

        timestamp_df = final_df.withColumn(
            "timestamp", (f.col("start_hour") * 60 + f.col("start_min")) * 60
        )

        for row in timestamp_df.collect():
            # Output to file
            output_file = os.path.join(output_path, f"part-{row['timestamp']}")
            timestamp_df.filter(
                (f.col("start_hour") == row["start_hour"])
                & (f.col("start_min") == row["start_min"])
            ).select("trend").write.mode("overwrite").text(output_file)

            # Stdout
            headquarter = row["headquarter"]
            current_count = row["count"]
            previous_count = row["prev_count"]
            timestamp = row["timestamp"]
            if headquarter == "goldman":
                print(
                    f"The number of arrivals to Goldman Sachs has doubled from {previous_count} to {current_count} at {timestamp}!"
                )
            elif headquarter == "citigroup":
                print(
                    f"The number of arrivals to Citigroup has doubled from {previous_count} to {current_count} at {timestamp}!"
                )

    return batch_func


def main(input_path, checkpoint_path, output_path):
    # Creating a SparkSession in Python
    spark = SparkSession.builder.master("local").appName("Spark EX4").getOrCreate()

    # Configures the number of partitions to use when shuffling data for joins or aggregations.
    spark.conf.set("spark.sql.shuffle.partitions", "10")

    # Stream for reading
    full_df = spark.readStream.option("maxFilesPerTrigger", step_files).csv(
        input_path, header=False, schema=full_schema
    )

    yellow_df = full_df.where(full_df["_c0"] == "yellow").selectExpr(
        *[
            f"cast(_c{i} as {field.dataType.simpleString()}) as {field.name}"
            for i, field in enumerate(yellow_schema.fields)
        ]
    )

    green_df = full_df.where(full_df["_c0"] == "green").selectExpr(
        *[
            f"cast(_c{i} as {field.dataType.simpleString()}) as {field.name}"
            for i, field in enumerate(green_schema.fields)
        ]
    )

    unified_df = yellow_df.unionByName(green_df, allowMissingColumns=True)

    # Create column to mark
    processed_df = unified_df.withColumn("goldman", in_polygon("Dropoff_longitude", "Dropoff_latitude", f.lit(goldman)))\
        .withColumn("citigroup", in_polygon("Dropoff_longitude", "Dropoff_latitude", f.lit(citigroup)))

    processed_df = processed_df.where("goldman == 1 OR citigroup == 1")
    processed_df = processed_df.withColumn(
        "headquarter", f.when(f.col("goldman") == 1, "goldman").otherwise("citigroup")
    )

    by_dropoff = processed_df.groupBy(
        f.window(f.col("Dropoff_datetime"), "10 minutes"), 
        "headquarter"
    ).count()

    query = (
        by_dropoff.writeStream.outputMode("complete")
        .foreachBatch(foreach_batch_function(output_path))
        .queryName("TrendingArrivals")
        .option("checkpointLocation", checkpoint_path)
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spark Job with input, checkpoint, and output arguments"
    )
    parser.add_argument("--input", required=True, help="Path to input data")
    parser.add_argument(
        "--checkpoint",
        default="/tmp/checkpoint",
        required=False,
        help="Path to checkpoint data",
    )
    parser.add_argument("--output", required=True, help="Path to output data")

    args = parser.parse_args()

    findspark.init()
    main(args.input, args.checkpoint, args.output)
