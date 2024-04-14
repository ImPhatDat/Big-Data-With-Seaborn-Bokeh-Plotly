import java.util.*;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.BufferedReader;

import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class TextCleaning {
    // Load stopwords, bbc.terms, and bbc.docs in the driver class
    private static Set<String> stopWords = new HashSet<>();
    private static List<String> termIDs = new ArrayList<>();
    private static List<String> docIDs = new ArrayList<>();

    private static void loadHelpers(FileSystem fs,
            String stopWordsFilePath, String termFilePath, String docFilePath) throws IOException {
        String line;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(stopWordsFilePath))))) {
            while ((line = br.readLine()) != null) {
                stopWords.add(line.trim());
            }
        }

        try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(termFilePath))))) {
            while ((line = br.readLine()) != null) {
                termIDs.add(line.trim());
            }
        }

        try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(docFilePath))))) {
            while ((line = br.readLine()) != null) {
                docIDs.add(line.trim());
            }
        }
    }

    public static class Prepare extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);

        private String removePunctuation(String token) {
            return token.replaceAll("[^a-zA-Z0-9£]", "");
        }

        private String getFolderName(Context context) {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            Path path = fileSplit.getPath();
            return path.getParent().getName();
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String folderName = getFolderName(context);
            String fileName = fileSplit.getPath().getName().split("\\.")[0];
            String line = value.toString().toLowerCase();
            StringTokenizer tokenizer = new StringTokenizer(line);

            while (tokenizer.hasMoreTokens()) {
                String token = tokenizer.nextToken();
                token = removePunctuation(token);

                if (!token.isEmpty() && !stopWords.contains(token)) {
                    context.write(new Text(folderName + "." + fileName + "_" + token), one);
                }
            }
        }
    }

    public static class Process extends Reducer<Text, IntWritable, Text, Text> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int count = 0;
            int termID = getTermID(key);
            int docID = getDocID(key);

            for (IntWritable val : values) {
                count += val.get();
            }
            if (termID != 0 && docID != 0) {
                context.write(new Text(String.valueOf(termID)),
                        new Text(String.valueOf(docID) + "\t" + String.valueOf(count)));
            }
        }

        private int getDocID(Text key) {
            String[] keyString = key.toString().split("_");
            String file = keyString[0];
            return docIDs.indexOf(file) + 1;
        }

        private int getTermID(Text key) {
            String[] keyString = key.toString().split("_");
            String word = keyString[1];
            return termIDs.indexOf(word) + 1;
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        // Load stopwords, bbc.terms, and bbc.docs
        FileSystem fs = FileSystem.get(conf);
        loadHelpers(
            fs,
            "/helpers/stopwords.txt",
            "/helpers/bbc.terms",
            "/helpers/bbc.docs"
        );

        // Pass loaded contents to configuration
        conf.set("stopWords", String.join(",", stopWords));
        conf.set("termIDs", String.join(",", termIDs));
        conf.set("docIDs", String.join(",", docIDs));

        Job job = Job.getInstance(conf, "Text Cleaning");
        if (fs.exists(new Path(args[1])))
            fs.delete(new Path(args[1]), true);

        job.setJarByClass(TextCleaning.class);
        job.setMapperClass(Prepare.class);
        job.setReducerClass(Process.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
