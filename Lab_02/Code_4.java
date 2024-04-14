import java.util.*;

import java.io.IOException;

import java.io.InputStreamReader;
import java.io.BufferedReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.LongWritable;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.fs.FileContext;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;

import org.apache.hadoop.fs.CreateFlag;

public class Code_4 {
    // Calculate TF
    public static class MapJob1 extends Mapper<Object, Text, Text, Text> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] token = value.toString().split("\\s+");
            context.write(new Text(token[1]), new Text(token[0] + "\t" + token[2]));
        }
    }

    public static class ReduceJob1 extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            double totalWords = 0;
            List<String> temp = new ArrayList<>();

            for (Text val : values) {
                temp.add(val.toString());
                String[] tokens = val.toString().split("\\s+");
                totalWords += Integer.valueOf(tokens[1]);
            }

            for (String term : temp) {
                String[] tokens = term.split("\\s+");
                double TF = Double.valueOf(tokens[1]) / totalWords;
                context.write(new Text(tokens[0] + "\t" + key.toString() + "\t" + Double.toString(TF)), new Text(" "));
            }
        }
    }

    // Calculate IDF
    public static class MapJob2 extends Mapper<Object, Text, Text, Text> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] token = value.toString().split("\\s+");
            context.write(new Text(token[0]), new Text(token[1] + "\t" + token[2]));
        }
    }

    // term (doc + freq)
    public static class ReduceJob2 extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            double totalDoc = context.getConfiguration().getDouble("totalDoc", -1);
            List<String> temp = new ArrayList<String>();
            double totalDocTerm = 0;

            for (Text val : values) {
                temp.add(val.toString());
                totalDocTerm += 1;
            }
            double IDF = Math.log(totalDoc / totalDocTerm);
            for (String term : temp) {
                String[] tokens = term.toString().split("\\s+");
                double TF = Double.valueOf(tokens[1]);
                double result = TF * IDF;
                context.write(new Text(tokens[0] + "\t" + key.toString() + "\t" +
                        Double.toString(result)),
                        new Text(" "));
            }
        }
    }

    public static class DocMap extends Mapper<Object, Text, Text, Text> {
        HashSet<String> docID = new HashSet<>();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split("\\s+");
            if (!docID.contains(tokens[1])) {
                docID.add(tokens[1]);
                context.write(new Text(" "), new Text(tokens[1]));
            }
        }
    }

    public static class DocReduce extends Reducer<Text, Text, LongWritable, NullWritable> {
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            long count = 0;
            for (Text value : values) {
                count++;
            }
            context.write(new LongWritable(count), NullWritable.get());
        }
    }

    public static void writeReducerOutput(FileSystem fs, Path result, Path out) throws IOException {
        FileStatus[] fStatuses = fs.listStatus(result);
        FileContext fileContext = FileContext.getFileContext();
        FSDataOutputStream outStream = fileContext.create(out, EnumSet.of(CreateFlag.CREATE, CreateFlag.OVERWRITE));

        for (FileStatus status : fStatuses) {
            String line = new String();
            FSDataInputStream reduceOutputStream = fileContext.open(status.getPath());
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(reduceOutputStream));
            while ((line = bufferedReader.readLine()) != null) {
                outStream.writeBytes(line + "\n");
            }
            bufferedReader.close();
        }
        outStream.close();
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        if (fs.exists(new Path("./tempDir")))
            fs.delete(new Path("./tempDir"), true);
        if (fs.exists(new Path(args[1])))
            fs.delete(new Path(args[1]), true);

        // Set total doc
        Job jobTotal = Job.getInstance(conf, "totaldoc");
        FileInputFormat.addInputPath(jobTotal, new Path(args[0]));
        FileOutputFormat.setOutputPath(jobTotal, new Path("./tempDir/sum"));

        jobTotal.setJarByClass(Code_4.class);
        jobTotal.setMapperClass(DocMap.class);
        jobTotal.setMapOutputKeyClass(Text.class);
        jobTotal.setMapOutputValueClass(Text.class);

        jobTotal.setReducerClass(DocReduce.class);
        jobTotal.setOutputKeyClass(LongWritable.class);
        jobTotal.setOutputValueClass(NullWritable.class);

        // Đợi job Total xong
        boolean success = jobTotal.waitForCompletion(true);
        if (!success) {
            System.exit(1);
        }

        if (fs.exists(new Path("./tempDir/sum/SUCCESS"))) {
            Path outputPath = new Path("./tempDir/sum/part-r-00000");
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(outputPath)))) {
                String line;
                while ((line = br.readLine()) != null) {
                    double totaldocs = Double.parseDouble(line.strip());
                    conf.setDouble("totalDoc", totaldocs);
                    br.close();
                    break;
                }
            }
        }

        // Job 1
        Job job1 = Job.getInstance(conf, "TF");
        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, new Path("./tempDir/cal"));

        job1.setJarByClass(Code_4.class);
        job1.setMapperClass(MapJob1.class);
        job1.setReducerClass(ReduceJob1.class);

        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);

        // Đợi job Total xong
        success = job1.waitForCompletion(true);
        if (!success) {
            System.exit(1);
        }

        // Job 2
        Job job2 = Job.getInstance(conf, "IDF");
        FileInputFormat.addInputPath(job2, new Path("./tempDir/cal"));
        FileOutputFormat.setOutputPath(job2, new Path(args[1]));

        job2.setJarByClass(Code_4.class);
        job2.setMapperClass(MapJob2.class);
        job2.setReducerClass(ReduceJob2.class);

        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        success = job2.waitForCompletion(true);
        if (success) {
            writeReducerOutput(fs, new Path(args[1]), new Path("task_1_4.mtx"));
        }
        System.exit(1);
    }
}