import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object StreamTest {
    def main(args: Array[String]): Unit = {

    }
}

val dummySchema = StructType((0 until 22).map(i => StructField(s"_c$i", StringType, nullable = true)))

// define schema for yellow taxi trips
val yellow_taxi_schema = StructType(Array(
    StructField("type", StringType(), nullable=False),
    StructField("VendorID", IntegerType()),
    StructField("tpep_pickup_datetime", TimestampType()),
    StructField("tpep_dropoff_datetime", TimestampType()),
    StructField("passenger_count", IntegerType()),
    StructField("trip_distance", FloatType()),
    StructField("pickup_longitude", FloatType()),
    StructField("pickup_latitude", FloatType()),
    StructField("RatecodeID", FloatType()),
    StructField("store_and_fwd_flag", StringType()),
    StructField("dropoff_longitude", FloatType()),
    StructField("dropoff_latitude", FloatType()),
    StructField("payment_type", IntegerType()),
    StructField("fare_amount", FloatType()),
    StructField("extra", FloatType()),
    StructField("mta_tax", FloatType()),
    StructField("tip_amount", FloatType()),
    StructField("tolls_amount", FloatType()),
    StructField("improvement_surcharge", FloatType()),
    StructField("total_amount", FloatType())
))
// define schema for green taxi trips
val green_taxi_schema = StructType(Array(
    StructField("type", StringType(), nullable=False),
    StructField("VendorID", IntegerType()),
    StructField("lpep_pickup_datetime", TimestampType()),
    StructField("Lpep_dropoff_datetime", TimestampType()),
    StructField("Store_and_fwd_flag", StringType()),
    StructField("RateCodeID", IntegerType()),
    StructField("Pickup_longitude", FloatType()),
    StructField("Pickup_latitude", FloatType()),
    StructField("Dropoff_longitude", FloatType()),
    StructField("Dropoff_latitude", FloatType()),
    StructField("Passenger_count", IntegerType()),
    StructField("Trip_distance", FloatType()),
    StructField("Fare_amount", FloatType()),
    StructField("Extra", FloatType()),
    StructField("MTA_tax", FloatType()),
    StructField("Tip_amount", FloatType()),
    StructField("Tolls_amount", FloatType()),
    StructField("Ehail_fee", FloatType()),
    StructField("improvement_surcharge", FloatType()),
    StructField("Total_amount", FloatType()),
    StructField("Payment_type", IntegerType()),
    StructField("Trip_type", IntegerType())
))

def main(input_path):
    global dummy_schema, yellow_taxi_schema, green_taxi_schema
    
    findspark.init()
    # Creating a SparkSession in Python
    spark = SparkSession.builder\
        .master("local")\
        .appName("Spark Streaming Test")\
        .getOrCreate()
    # keep the size of shuffles small
    spark.conf.set("spark.sql.shuffle.partitions", "2")
    
    # read the entire data files first
    full_df = spark.readStream\
        .option("maxFilesPerTrigger", 100)\
        .csv(f"file://{input_path}",
            header=False,
            schema=dummy_schema)

    # Convert to schema based on the 'type' column
    yellow_df = full_df.where(full_df["_c0"] == "yellow").selectExpr(
        *[f"cast(_c{i} as {field.dataType.simpleString()}) as {field.name}" \
            for i, field in enumerate(yellow_taxi_schema.fields)]
    )

    green_df = full_df.where(full_df["_c0"] == "green").selectExpr(
        *[f"cast(_c{i} as {field.dataType.simpleString()}) as {field.name}" \
            for i, field in enumerate(green_taxi_schema.fields)]
    )
    
    # Process streaming data: Type (yellow/green) count
    sub_yellow = yellow_df.select("type")
    sub_green = green_df.select("type")
    
    sub_df = sub_yellow.union(sub_green)
    
    by_type = sub_df.groupBy("type").count()

    # Query
    query = by_type.writeStream\
        .format("console")\
        .outputMode("complete")\
        .queryName("Type_Count")\
        .option("truncate", "false")\
        .start()
    
    query.awaitTermination()
    
    query.stop()
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark Job with input argument")
    parser.add_argument("--input", required=True, help="Path to input data")
    args = parser.parse_args()
    
    main(args.input)