import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object EventCount {
    def main(args: Array[String]): Unit = {
        if (args.length != 2 && args.length != 3) {
            println(
              "Usage: EventCount <inputPath> [<checkpointPath>] <outputPath>"
            )
            sys.exit(1)
        }

        var inputPath = args(0)
        var outputPath = args(1)
        var checkpointPath = "checkpoint"
        if (args.length == 3) {
            checkpointPath = args(1)
            outputPath = args(2)
        }
        val cwd = System.getProperty("user.dir")
        inputPath = cwd + "/" + inputPath
        outputPath = cwd + "/" + outputPath
        checkpointPath = cwd + "/" + checkpointPath

        // Creating a SparkSession
        val spark = SparkSession.builder
            .master("local")
            .appName("Spark Streaming Test")
            .getOrCreate()

        // dummy schema for reading all rows
        val dummy_schema = StructType(
          (0 until 22).map(i =>
              StructField(s"_c$i", StringType, nullable = true)
          )
        )
        // define schema for yellow taxi trips
        val yellow_taxi_schema = StructType(
          Array(
            StructField("type", StringType, nullable = false),
            StructField("VendorID", IntegerType),
            StructField("tpep_pickup_datetime", TimestampType),
            StructField("tpep_dropoff_datetime", TimestampType),
            StructField("passenger_count", IntegerType),
            StructField("trip_distance", FloatType),
            StructField("pickup_longitude", FloatType),
            StructField("pickup_latitude", FloatType),
            StructField("RatecodeID", FloatType),
            StructField("store_and_fwd_flag", StringType),
            StructField("dropoff_longitude", FloatType),
            StructField("dropoff_latitude", FloatType),
            StructField("payment_type", IntegerType),
            StructField("fare_amount", FloatType),
            StructField("extra", FloatType),
            StructField("mta_tax", FloatType),
            StructField("tip_amount", FloatType),
            StructField("tolls_amount", FloatType),
            StructField("improvement_surcharge", FloatType),
            StructField("total_amount", FloatType)
          )
        )
        // define schema for green taxi trips
        val green_taxi_schema = StructType(
          Array(
            StructField("type", StringType, nullable = false),
            StructField("VendorID", IntegerType),
            StructField("lpep_pickup_datetime", TimestampType),
            StructField("Lpep_dropoff_datetime", TimestampType),
            StructField("Store_and_fwd_flag", StringType),
            StructField("RateCodeID", IntegerType),
            StructField("Pickup_longitude", FloatType),
            StructField("Pickup_latitude", FloatType),
            StructField("Dropoff_longitude", FloatType),
            StructField("Dropoff_latitude", FloatType),
            StructField("Passenger_count", IntegerType),
            StructField("Trip_distance", FloatType),
            StructField("Fare_amount", FloatType),
            StructField("Extra", FloatType),
            StructField("MTA_tax", FloatType),
            StructField("Tip_amount", FloatType),
            StructField("Tolls_amount", FloatType),
            StructField("Ehail_fee", FloatType),
            StructField("improvement_surcharge", FloatType),
            StructField("Total_amount", FloatType),
            StructField("Payment_type", IntegerType),
            StructField("Trip_type", IntegerType)
          )
        )

        // read the entire data files first
        val df = spark.readStream
            .format("csv")
            .option("maxFilesPerTrigger", 100)
            .option("header", false)
            .schema(dummy_schema)
            .load(s"file://$inputPath")

        // Convert to schema based on the 'type' column
        val yellow_df = df
            .where(col("_c0") === "yellow")
            .selectExpr(
              yellow_taxi_schema.fields.zipWithIndex.map { case (field, i) =>
                  s"cast(_c$i as ${field.dataType.simpleString}) as ${field.name}"
              }: _*
            )
        val green_df = df
            .where(col("_c0") === "green")
            .selectExpr(
              green_taxi_schema.fields.zipWithIndex.map { case (field, i) =>
                  s"cast(_c$i as ${field.dataType.simpleString}) as ${field.name}"
              }: _*
            )

        // Process streaming data: Event count
        val sub_yellow = yellow_df.selectExpr("type", "tpep_dropoff_datetime as dropoff_datetime")
        val sub_green = green_df.selectExpr("type", "Lpep_dropoff_datetime as dropoff_datetime")

        val sub_df = sub_yellow.union(sub_green)

        val by_dropoff = sub_df
            .groupBy(window(col("dropoff_datetime"), "1 hour"))
            .count()

        // Query and write to output file
        val query = by_dropoff.writeStream
            .queryName("Event_Count")
            .outputMode("update")
            .option("checkpointLocation", s"file://$checkpointPath")
            .foreachBatch{(batch_df: DataFrame, batchId: Long) => {
                // get start hour column
                val tmp_df = batch_df.withColumn("startHour", hour(col("window.start")))
                
                // for each start hour, write to different directory
                val start_hours = tmp_df.select("startHour").distinct().collect()
                
                if (start_hours.length != 0) {
                    start_hours.foreach { start_hour =>
                        val hNum = start_hour.getAs[Int]("startHour")
                        val sub_df = tmp_df.where(col("startHour") === lit(hNum)).select("count")
                        val outputDir = s"file://$outputPath/output-${(hNum + 1) * 60 * 60 * 1000}"
                        
                        sub_df.write.mode("overwrite").json(outputDir)
                    }
                }
            }}
            .start()

        query.awaitTermination()

        query.stop()
        spark.stop()
    }
}
