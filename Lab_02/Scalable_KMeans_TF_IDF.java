import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Collections;
import java.util.Comparator;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.conf.Configuration;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Scalable_KMeans_TF_IDF {
    public static class GetTerms extends Mapper<Object, Text, Text, IntWritable> {
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            try {
                String[] parts = value.toString().trim().split("\\t");
                context.write(new Text(" "), new IntWritable(Integer.parseInt(parts[0])));
            } catch (Exception e) {
                return; // ignore lines where we can't parse it
            }
        }
    }

    public static class GetUniqueTerms extends Reducer<Text, IntWritable, IntWritable, NullWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            Set<Integer> docs = new HashSet<>();
            for (IntWritable val : values) {
                int termid = val.get();
                if (!docs.contains(termid)) {
                    docs.add(termid);
                }
            }
            List<Integer> sortedList = new ArrayList<>(docs);
            Collections.sort(sortedList);
            for (Integer termId : sortedList) {
                context.write(new IntWritable(termId), NullWritable.get());
            }
        }
    }

    public static List<String> saveTermToConf(Configuration conf, FileSystem fs, String inputPath) throws IOException {
        FileStatus[] files = fs.listStatus(new Path(inputPath));
        String output = "";
        List<String> terms = new ArrayList<>();
        int i = 0;
        for (FileStatus file : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(file.getPath())))) {
                String line;
                while ((line = br.readLine()) != null) {
                    output += line + ",";
                    terms.add(i, line);
                }
                br.close();
            }
        }

        output = output.substring(0, output.length() - 2);
        conf.set("uniqueTerms", output);
        return terms;
    }

    public static List<String> readTermFromConf(Configuration conf) {
        List<String> terms = new ArrayList<>();
        String[] all_terms = conf.get("uniqueTerms").split(",");
        for (int i = 0; i < all_terms.length; i++) {
            terms.add(i, all_terms[i]);
        }
        return terms;
    }

    public static class SwapTermDoc extends Mapper<Object, Text, IntWritable, Text> {
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().trim().split("\\t");
            context.write(new IntWritable(Integer.parseInt(parts[1])), new Text(parts[0] + ":" + parts[2]));
        }
    }

    public static class AggregateDoc extends Reducer<IntWritable, Text, Text, NullWritable> {
        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            String output = key.toString() + "|";
            for (Text val : values) {
                output += val.toString() + ",";
            }
            output = output.substring(0, output.length() - 2);
            context.write(new Text(output), NullWritable.get());
        }
    }

    public static class TF_IDF_Vector {
        private String docId;
        private double magnitude;
        private List<String> termList;
        private List<Double> values;
        private String repString;

        public TF_IDF_Vector() {
            this.docId = "";
            this.repString = "";
            this.magnitude = 0;
            this.termList = new ArrayList<>();
            this.values = new ArrayList<>();
        }

        public TF_IDF_Vector(List<String> terms) {
            this.docId = "0";
            this.repString = "";
            this.magnitude = 0;
            this.termList = new ArrayList<>(terms.size());
            this.values = new ArrayList<>(terms.size());
            for (int i = 0; i < terms.size(); i++) {
                this.termList.add(i, terms.get(i));
                this.values.add(i, 0.0);
            }
        }

        // Copy constructor
        public TF_IDF_Vector(TF_IDF_Vector original) {
            this.docId = original.docId;
            this.repString = original.repString;
            this.magnitude = original.magnitude;
            this.termList = new ArrayList<>(original.termList);
            this.values = new ArrayList<>(original.values);
        }

        public TF_IDF_Vector(String repString) {
            this();
            try {
                this.repString = repString;
                String[] docterm = repString.split("\\|");
                this.docId = docterm[0];

                String[] parts = docterm[1].split(",");

                double sumOfSquares = 0.0;
                for (int i = 0; i < parts.length; i++) {
                    String[] subparts = parts[i].split(":");

                    this.termList.add(subparts[0]);
                    double tfidf = Double.parseDouble(subparts[1]);
                    this.values.add(tfidf);
                    sumOfSquares += Math.pow(tfidf, 2);
                }
                this.magnitude = Math.sqrt(sumOfSquares);
            } catch (Exception e) {
                // System.err.println("Error parsing TF-IDF values: " + e.getMessage());
                return;
            }
        }

        // Copy function
        public TF_IDF_Vector copy() {
            return new TF_IDF_Vector(this);
        }

        public String getDocId() {
            return this.docId;
        }

        public void setDocId(String docId) {
            this.docId = docId;
        }

        public String getString() {
            return this.repString;
        }

        public List<String> getTerms() {
            return this.termList;
        }

        public List<Double> getValues() {
            return this.values;
        }

        public void setValue(int index, double newval) {
            this.values.set(index, newval);
        }

        public double getMagnitude() {
            return this.magnitude;
        }

        public double updateMagnitude() {
            double sumOfSquares = 0.0;
            for (int i = 0; i < this.getLength(); i++) {
                sumOfSquares += Math.pow(this.getTfIdf(i), 2);
            }
            this.magnitude = Math.sqrt(sumOfSquares);
            return this.magnitude;
        }

        public String getTerm(int index) {
            return this.termList.get(index);
        }

        public double getTfIdf(int index) {
            return this.values.get(index);
        }

        public int findTerm(String term) {
            return this.termList.indexOf(term);
        }

        public int getLength() {
            return this.termList.size();
        }

        public String updateString() {
            String output = this.docId + "|";
            for (int i = 0; i < this.getLength(); i++) {
                output += this.getTerm(i) + ":" + this.getTfIdf(i);
                if (i != this.getLength() - 1) {
                    output += ",";
                }
            }
            this.repString = output;
            return output;
        }
    }

    // define distance function
    public static double calculate_distance_cosine(TF_IDF_Vector vector1, TF_IDF_Vector vector2) {
        // Compute magnitudes
        double magnitude1 = vector1.updateMagnitude();
        double magnitude2 = vector2.updateMagnitude();

        if (magnitude1 == 0 || magnitude2 == 0) {
            return Double.MAX_VALUE; // Avoid division by zero
        }

        // Compute dot product
        double dotProduct = 0.0;

        for (int i = 0; i < vector1.getLength(); i++) {
            int termPos = vector2.findTerm(vector1.getTerm(i));
            if (termPos != -1) {
                // we don't need to care about the case where a term exists in a vector but not
                // in the other
                // since multiply by 0 is 0 (we assume if a term don't exists in a vector has
                // tf-idf value of 0)
                dotProduct += vector2.getTfIdf(termPos) * vector1.getTfIdf(i);
            }
        }

        // Compute cosine similarity
        return 1 - (dotProduct / (magnitude1 * magnitude2));
    }

    // Euclidean distance
    public static double calculate_distance_euclid(TF_IDF_Vector centroid, TF_IDF_Vector vector) {
        // Compute dot product
        double sumOfSquares = 0.0;

        // we know that centroid length is always larger or equal than any other TF-IDF
        // vector
        // since a centroid contains all terms
        for (int i = 0; i < centroid.getLength(); i++) {
            int termPos = vector.findTerm(centroid.getTerm(i));
            if (termPos != -1) {
                sumOfSquares += Math.pow(vector.getTfIdf(termPos) - centroid.getTfIdf(i), 2);
            } else {
                sumOfSquares += Math.pow(centroid.getTfIdf(i), 2);
            }
        }
        return Math.sqrt(sumOfSquares);
    }

    // get centroid (mean of all terms)
    public static TF_IDF_Vector getCentroid(List<TF_IDF_Vector> vectors, List<String> terms) {
        TF_IDF_Vector centroid = new TF_IDF_Vector(terms);

        for (int iterm = 0; iterm < terms.size(); iterm++) {
            double sum = 0;
            int count = 0;
            for (int j = 0; j < vectors.size(); j++) {
                TF_IDF_Vector current = vectors.get(j);
                int pos = current.findTerm(terms.get(iterm));
                if (pos != -1) {
                    sum += current.getTfIdf(pos);
                    count++;
                }
            }
            if (count != 0)
                centroid.setValue(iterm, sum / count);
        }

        centroid.updateString();
        return centroid;
    }

    // wheel selection
    private static TF_IDF_Vector selectRandomWeightedCentroid(List<Integer> w, List<TF_IDF_Vector> candidates) {
        int totalWeight = 0;
        for (int i = 0; i < candidates.size(); i++) {
            totalWeight += w.get(i);
        }
        int randomWeight = (int) (Math.random() * totalWeight);
        int cumulativeWeight = 0;
        for (int i = 0; i < candidates.size(); i++) {
            cumulativeWeight += w.get(i);
            if (cumulativeWeight > randomWeight) {
                return candidates.get(i);
            }
        }
        // return the last centroid if the above case don't happen
        return candidates.get(candidates.size() - 1);
    }
    

    // Unclustered
    public static List<TF_IDF_Vector> initializeCentroids(int l, int k, FileSystem fs, String inputPath,
            List<String> terms)
            throws IOException {
        List<TF_IDF_Vector> vectors = new ArrayList<>();

        int ic = 0;
        FileStatus[] files = fs.listStatus(new Path(inputPath));
        for (FileStatus file : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(file.getPath())))) {
                String line;
                while ((line = br.readLine()) != null) {
                    TF_IDF_Vector currentVector = new TF_IDF_Vector(line);
                    if (currentVector.docId == "") { // parse failed
                        continue;
                    }
                    vectors.add(ic++, currentVector);
                }
                br.close();
            }
        }
        int choice = (int) (Math.random() * (vectors.size() + 1));
        TF_IDF_Vector initialCentroid = vectors.get(choice);

        List<TF_IDF_Vector> candidates = new ArrayList<>();
        candidates.add(initialCentroid);
        
        double potential = 0.0;
        List<Double> psi = new ArrayList<>();
        for (int i = 0; i < vectors.size(); i++) {
            double min_distance = Double.MAX_VALUE;
            for (TF_IDF_Vector centroid : candidates) {
                double distance = calculate_distance_cosine(centroid, vectors.get(i));
                if (distance < min_distance) {
                    min_distance = distance;
                }
            }
            psi.add(i, min_distance);
            potential += min_distance;
        }
        
        // sampling to select candidates
        for (int i = 0; i < Math.log(potential); i++) {
            for (int j = 0; j < vectors.size(); j++) {
                double prob = (l * psi.get(j)) / potential;
                if (Math.random() <= prob) {
                    candidates.add(vectors.get(j));

                    for (int e = 0; e < vectors.size(); e++) {
                        double min_distance = Double.MAX_VALUE;
                        for (TF_IDF_Vector centroid : candidates) {
                            double distance = calculate_distance_cosine(centroid, vectors.get(e));
                            if (distance < min_distance) {
                                min_distance = distance;
                            }
                        }
                        psi.set(e, min_distance);
                    }
                }
            }
        }

        // weight
        List<Integer> w = new ArrayList<>();
        for (int i = 0; i < candidates.size(); i++) {
            w.add(0);
        }
        for (int i = 0; i < vectors.size(); i++) {
            double min = calculate_distance_cosine(candidates.get(0), vectors.get(i));
            int index = 0;
            for (int j = 1; j < candidates.size(); j++) {
                double dis = calculate_distance_cosine(candidates.get(j), vectors.get(i));
                if (min > dis) {
                    min = dis;
                    index = j;
                }
            }
            w.set(index, 1 + w.get(index));
        }

        // select k centroid from candidates 
        List<TF_IDF_Vector> finalCentroid = new ArrayList<>();
        while (finalCentroid.size() != k) {
            TF_IDF_Vector selected = selectRandomWeightedCentroid(w, candidates);
            if (!finalCentroid.contains(selected))
                finalCentroid.add(selected);
        }
        return finalCentroid;
    }

    // Clustered (contained cluster num and centroid)
    public static List<TF_IDF_Vector> loadCentroids(FileSystem fs, String inputPath) throws IOException {
        List<TF_IDF_Vector> centroids = new ArrayList<>();

        FileStatus[] files = fs.listStatus(new Path(inputPath));
        for (FileStatus file : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(file.getPath())))) {
                String line;
                while ((line = br.readLine()) != null) {
                    try {
                        String[] parts = line.trim().split("\\t");
                        if (parts[0].charAt(0) == 'c') {
                            centroids.add(Integer.parseInt(parts[0].substring(1)) - 1, new TF_IDF_Vector(parts[1]));
                        }
                    } catch (Exception e) {
                        continue;
                    }
                }
                br.close();

            }
        }
        return centroids;
    }

    // Method to save centroids to configuration
    public static void saveCentroidsToConf(Configuration conf, List<TF_IDF_Vector> centroids) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < centroids.size(); i++) {
            sb.append(centroids.get(i).getString()).append("\n");
        }
        conf.set("centroids", sb.toString());
    }

    // Method to read centroids from configuration
    public static List<TF_IDF_Vector> readCentroidsFromConf(Configuration conf) {
        List<TF_IDF_Vector> centroids = new ArrayList<>();
        String centroidsStr = conf.get("centroids");
        if (centroidsStr != null && !centroidsStr.isEmpty()) {
            String[] lines = centroidsStr.split("\n");
            for (int i = 0; i < lines.length; i++) {
                centroids.add(i, new TF_IDF_Vector(lines[i]));
            }
        }
        return centroids;
    }

    public static class KmeanMapper extends Mapper<Object, Text, IntWritable, Text> {
        private List<TF_IDF_Vector> centroids;

        private IntWritable clus = new IntWritable();
        private Text coor = new Text();

        @Override
        public void setup(Context context) {
            centroids = readCentroidsFromConf(context.getConfiguration());
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts;
            TF_IDF_Vector vector;
            try {
                String sval = value.toString();
                if (sval.contains("\t")) {
                    parts = sval.trim().split("\t");
                    if (parts[0].charAt(0) == 'c') {
                        return;
                    }
                    vector = new TF_IDF_Vector(parts[1]);
                    if (vector.getDocId() == "") {
                        return;
                    }
                } else {
                    vector = new TF_IDF_Vector(sval.trim());
                    if (vector.getDocId() == "") {
                        return;
                    }
                }

            } catch (Exception e) {
                return;
            }

            int cluster_choice = 1;

            double min_distance = Double.MAX_VALUE;
            for (int i = 0; i < centroids.size(); i++) {
                double distance = calculate_distance_cosine(vector, centroids.get(i));
                if (distance < min_distance) {
                    min_distance = distance;
                    cluster_choice = i + 1;
                }
            }

            clus.set(cluster_choice);
            coor.set(vector.getString());
            context.write(clus, coor);
        }
    }

    public static class KmeanReducer extends Reducer<IntWritable, Text, Text, Text> {
        private List<String> terms;

        @Override
        public void setup(Context context) {
            terms = readTermFromConf(context.getConfiguration());
        }

        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            List<TF_IDF_Vector> vectors = new ArrayList<>();

            // Calculate the sum of all points assigned to this cluster
            for (Text value : values) {
                vectors.add(new TF_IDF_Vector(value.toString()));
                context.write(new Text(key.toString()), value);
            }

            TF_IDF_Vector centroid = getCentroid(vectors, terms);

            context.write(new Text("c" + key.toString()), new Text(centroid.getString()));
        }
    }

    public static class ValueTermPair {
        public double value; // use public for convenience
        public String term;

        public ValueTermPair(double value, String term) {
            this.value = value;
            this.term = term;
        }
    }

    public static void reportTopAndLoss(FileSystem fs, int k, String iterationsPath, String outputTopWords,
            String outputLoss) throws IOException, InterruptedException {
        String topWordString = "";
        String lossString = "";

        List<String> subDirectories = new ArrayList<>();

        FileStatus[] fileStatuses = fs.listStatus(new Path(iterationsPath));
        try {
            for (FileStatus status : fileStatuses) {
                if (status.isDirectory()) {
                    subDirectories.add(status.getPath().getName());
                }
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            return;
        }

        for (int pathi = 0; pathi < subDirectories.size(); pathi++) {
            String currentIteration = subDirectories.get(pathi);
            FileStatus[] files = fs.listStatus(new Path(iterationsPath + "/" + currentIteration));
            topWordString += currentIteration + "\n";
            lossString += currentIteration + "\n";

            List<List<TF_IDF_Vector>> vectors = new ArrayList<>(k);
            for (int i = 0; i < k; i++) {
                vectors.add(new ArrayList<>());
            }
            List<TF_IDF_Vector> centroids = new ArrayList<>();

            for (FileStatus file : files) {
                try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(file.getPath())))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        try {
                            String[] parts = line.trim().split("\\t");

                            TF_IDF_Vector current = new TF_IDF_Vector(parts[1]);
                            if (current.docId == "") {
                                continue;
                            }
                            int cluster = 1;
                            if (parts[0].charAt(0) == 'c') {
                                cluster = Integer.parseInt(parts[0].substring(1)) - 1;
                                centroids.add(cluster, current);
                            } else {
                                cluster = Integer.parseInt(parts[0]) - 1;
                                vectors.get(cluster).add(current);
                            }

                        } catch (Exception e) {
                            continue;
                        }
                    }
                    br.close();
                }
            }

            for (int i = 0; i < centroids.size(); i++) {
                topWordString += (i + 1) + "\t";

                List<Double> sortedValues = centroids.get(i).getValues();
                List<String> sortedTerms = centroids.get(i).getTerms();

                // Create a list of pairs (value, term)
                List<ValueTermPair> pairs = new ArrayList<>();
                for (int j = 0; j < sortedValues.size(); j++) {
                    pairs.add(new ValueTermPair(sortedValues.get(j), sortedTerms.get(j)));
                }

                // Sort the list of pairs based on values
                Collections.sort(pairs, Comparator.comparingDouble((ValueTermPair pair) -> pair.value).reversed());

                final int top = 10;
                for (int j = 0; j < top; j++) {
                    topWordString += pairs.get(j).term + ":" + pairs.get(j).value;
                    if (j != top - 1) {
                        topWordString += ",";
                    }
                }
                topWordString += "\n";

            }

            double loss = 0.0;
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < vectors.get(i).size(); j++) {
                    loss += calculate_distance_euclid(centroids.get(i), vectors.get(i).get(j));
                }
            }
            lossString += loss + "\n";
        }

        // write to top word file
        OutputStream os = fs.create(new Path(outputTopWords));
        BufferedWriter br = new BufferedWriter(new OutputStreamWriter(os));

        br.write(topWordString);
        br.close();

        // write to loss file
        os = fs.create(new Path(outputLoss));
        br = new BufferedWriter(new OutputStreamWriter(os));

        br.write(lossString);
        br.close();

    }

    public static void writeToFinalOutput(FileSystem fs, int k, String finalInputPath, String iterationPath,
            String outputCluster, String outputClasses, String outputTopWords, String outputLoss)
            throws IOException, InterruptedException {

        List<List<TF_IDF_Vector>> vectors = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            vectors.add(new ArrayList<>());
        }
        List<TF_IDF_Vector> centroids = new ArrayList<>();

        FileStatus[] files = fs.listStatus(new Path(finalInputPath));
        for (FileStatus file : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(file.getPath())))) {
                String line;
                while ((line = br.readLine()) != null) {
                    try {
                        String[] parts = line.trim().split("\\t");

                        TF_IDF_Vector current = new TF_IDF_Vector(parts[1]);
                        if (current.docId == "") {
                            continue;
                        }
                        int cluster = 1;
                        if (parts[0].charAt(0) == 'c') {
                            cluster = Integer.parseInt(parts[0].substring(1)) - 1;
                            centroids.add(cluster, current);
                        } else {
                            cluster = Integer.parseInt(parts[0]) - 1;
                            vectors.get(cluster).add(current);
                        }

                    } catch (Exception e) {
                        continue;
                    }
                }
                br.close();
            }
        }

        // write to cluster centers file
        OutputStream os = fs.create(new Path(outputCluster));

        BufferedWriter br = new BufferedWriter(new OutputStreamWriter(os));
        for (int i = 0; i < k; i++) {
            TF_IDF_Vector centroid = centroids.get(i);
            br.write((i + 1) + "\t" + centroid.getString());
            br.newLine();
        }
        br.close();

        // write to classes file
        os = fs.create(new Path(outputClasses));

        br = new BufferedWriter(new OutputStreamWriter(os));
        for (int i = 0; i < vectors.size(); i++) {
            for (int j = 0; j < vectors.get(i).size(); j++) {
                TF_IDF_Vector vector = vectors.get(i).get(j);
                br.write((i + 1) + "\t" + vector.getString());
                br.newLine();
            }
        }
        br.close();

        reportTopAndLoss(fs, k, iterationPath, outputTopWords, outputLoss);
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2 && args.length != 4) {
            System.err.println("Usage: [k maxIterations] inputPath outputPath");
            System.exit(1); // Exit with an error code
        }
        final double THRESHOLD = 0.00001;
        int k = 5;
        int maxIterations = 10;
        String inputPathS = args[0];
        String outputPathS = args[1];
        
        if (args.length == 4) {
            k = Integer.parseInt(args[0]);
            maxIterations = Integer.parseInt(args[1]);
            inputPathS = args[2];
            outputPathS = args[3];
        }
        int l = 4 * k;
        
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        Job job = Job.getInstance(conf, "Get unique terms");

        job.setJarByClass(Scalable_KMeans_TF_IDF.class);

        job.setMapperClass(GetTerms.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setReducerClass(GetUniqueTerms.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(NullWritable.class);

        FileInputFormat.addInputPath(job, new Path(inputPathS));
        FileOutputFormat.setOutputPath(job, new Path(outputPathS));

        if (!job.waitForCompletion(true)) {
            System.exit(1);
        }

        if (!fs.exists(new Path(outputPathS + "/_SUCCESS"))) {
            System.exit(1);
        }

        List<String> terms = saveTermToConf(conf, fs, outputPathS);

        fs.delete(new Path(outputPathS), true);

        job = Job.getInstance(conf, "Transform input");

        job.setJarByClass(Scalable_KMeans_TF_IDF.class);

        job.setMapperClass(SwapTermDoc.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);

        job.setReducerClass(AggregateDoc.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);

        FileInputFormat.addInputPath(job, new Path(inputPathS));
        FileOutputFormat.setOutputPath(job, new Path(outputPathS));

        if (!job.waitForCompletion(true)) {
            System.exit(1);
        }

        String transformed_file = "./tfidf.txt";

        if (!fs.exists(new Path(outputPathS + "/_SUCCESS"))) {
            System.exit(1);
        }

        if (fs.exists(new Path(transformed_file))) {
            fs.delete(new Path(transformed_file), true);
        }

        // Check if the move operation was successful
        if (!fs.rename(new Path(outputPathS + "/part-r-00000"), new Path(transformed_file))) {
            System.err.println("Failed to moved file.");
            System.exit(1);
        }
        fs.delete(new Path(outputPathS), true);

        List<TF_IDF_Vector> centroids = initializeCentroids(l, k, fs, transformed_file, terms);
        saveCentroidsToConf(conf, centroids);

        Path inputPath = new Path(transformed_file); // Initial input path
        Path outputPath;
        String finalPathS = "";
        String subPathS;
        for (int i = 0; i < maxIterations; i++) {
            job = Job.getInstance(conf, "K means TF-IDF - Iteration " + (i + 1));

            job.setJarByClass(Scalable_KMeans_TF_IDF.class);

            job.setMapperClass(KmeanMapper.class);
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Text.class);

            job.setReducerClass(KmeanReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            // Set input and output paths
            FileInputFormat.addInputPath(job, inputPath);
            outputPath = new Path(outputPathS + "/iteration_" + (i + 1));
            FileOutputFormat.setOutputPath(job, outputPath);

            job.waitForCompletion(true);

            // After each iteration, update centroids and set input path for next iteration
            subPathS = outputPathS + "/iteration_" + (i + 1);
            inputPath = new Path(subPathS + "/part-r-00000"); // Set input path for next iteration

            if (!fs.exists(new Path(subPathS + "/_SUCCESS"))) { // if one iteration failed, stop
                break;
            }

            List<TF_IDF_Vector> newCentroids = loadCentroids(fs, subPathS);
            // if the number of cluster change (empty cluster appears), stop
            if (newCentroids.size() != k) {
                finalPathS = outputPath.toString();
                break;
            }

            double maxDistance = Double.MIN_VALUE;
            for (int j = 0; j < centroids.size(); j++) {
                double c_distance = calculate_distance_cosine(centroids.get(j), newCentroids.get(j));
                if (c_distance > maxDistance) {
                    maxDistance = c_distance;
                }
            }

            if (maxDistance <= THRESHOLD || i == maxIterations - 1) {
                finalPathS = outputPath.toString();
                break;
            }

            centroids = newCentroids;
            saveCentroidsToConf(conf, centroids);
        }

        if (finalPathS != "") {
            // if all success
            writeToFinalOutput(
                    fs, k, finalPathS, outputPathS,
                    "./task_2_3.clusters",
                    "./task_2_3.classes",
                    "./task_2_3.txt",
                    "./task_2_3.loss");
        }

        fs.close();
        System.exit(0);
    }

}