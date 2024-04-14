import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

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

public class KMeans2D {

    public static class Point {
        private double x;
        private double y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        public double getX() {
            return x;
        }

        public void setX(double newx) {
            this.x = newx;
        }

        public double getY() {
            return y;
        }

        public void setY(double newy) {
            this.y = newy;
        }

    }

    // Unclustered
    public static List<Point> initializeCentroids(int k, FileSystem fs, String inputPath) throws IOException {
        List<List<Point>> points = new ArrayList<>(k);
        // Initialize each element in the list
        for (int i = 0; i < k; i++) {
            points.add(new ArrayList<>());
        }

        int empty_count = k;

        FileStatus[] files = fs.listStatus(new Path(inputPath));
        for (FileStatus file : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(file.getPath())))) {
                String line;
                while ((line = br.readLine()) != null) {
                    try {
                        String[] parts = line.trim().split(",");
                        double x, y;
                        x = Double.parseDouble(parts[1]);
                        y = Double.parseDouble(parts[2]);

                        if (empty_count != 0) {
                            empty_count--;
                            points.get(empty_count).add(new Point(x, y));
                        } else {
                            int clusterIndex = (int) (Math.random() * k);
                            points.get(clusterIndex).add(new Point(x, y));
                        }
                    } catch (Exception e) {
                        continue;
                    }
                }
                br.close();

            }
        }

        List<Point> centroids = new ArrayList<>();

        for (int i = 0; i < k; i++) {
            double sumX = 0;
            double sumY = 0;
            int count = points.get(i).size();
            for (int j = 0; j < points.get(i).size(); j++) {
                Point p = points.get(i).get(j);
                sumX += p.getX();
                sumY += p.getY();
            }
            centroids.add(i, new Point(sumX / count, sumY / count));
        }
        return centroids;
    }

    // Method to load centroids from a file
    public static List<Point> loadCentroids(int k, FileSystem fs, String inputPath) throws IOException {
        List<Point> centroids = new ArrayList<>();

        FileStatus[] files = fs.listStatus(new Path(inputPath));
        for (FileStatus file : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(file.getPath())))) {
                String line;
                while ((line = br.readLine()) != null) {
                    try {
                        String[] parts = line.trim().split(",");
                        if (parts[0].charAt(0) == 'c') {
                            double x = Double.parseDouble(parts[1]);
                            double y = Double.parseDouble(parts[2]);
                            centroids.add(Integer.parseInt(parts[0].substring(1)) - 1, new Point(x, y));
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
    public static void saveCentroidsToConf(Configuration conf, List<Point> centroids) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < centroids.size(); i++) {
            Point centroid = centroids.get(i);
            sb.append(centroid.getX()).append(",").append(centroid.getY()).append("\n");
        }
        conf.set("centroids", sb.toString());
    }

    // Method to read centroids from configuration
    public static List<Point> readCentroidsFromConf(Configuration conf) {
        List<Point> centroids = new ArrayList<>();
        String centroidsStr = conf.get("centroids");
        if (centroidsStr != null && !centroidsStr.isEmpty()) {
            String[] lines = centroidsStr.split("\n");
            for (int i = 0; i < lines.length; i++) {
                String[] parts = lines[i].split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                centroids.add(i, new Point(x, y));
            }
        }
        return centroids;
    }

    // define distance function
    public static double calculate_distance(Point p1, Point p2) {
        return Math.sqrt(Math.pow(p1.getX() - p2.getX(), 2) + Math.pow(p1.getY() - p2.getY(), 2));
    }

    public static class ClusterMapper extends Mapper<Object, Text, IntWritable, Text> {
        private List<Point> centroids;

        private IntWritable clus = new IntWritable();
        private Text coor = new Text();

        @Override
        public void setup(Context context) {
            centroids = readCentroidsFromConf(context.getConfiguration());
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts;
            Point point;

            try {
                parts = value.toString().trim().split(",");
                double x, y;
                x = Double.parseDouble(parts[1]);
                y = Double.parseDouble(parts[2]);
                if (parts[0].charAt(0) == 'c') {
                    return;
                }
                point = new Point(x, y);
            } catch (Exception e) {
                return;
            }

            int cluster_choice = 1;

            double min_distance = Double.MAX_VALUE;
            for (int i = 0; i < centroids.size(); i++) {
                double distance = calculate_distance(point, centroids.get(i));
                if (distance < min_distance) {
                    min_distance = distance;
                    cluster_choice = i + 1;
                }
            }

            clus.set(cluster_choice);
            coor.set(String.valueOf(point.getX()) + "," + String.valueOf(point.getY()));
            context.write(clus, coor);
        }
    }

    public static class ClusterReducer extends Reducer<IntWritable, Text, Text, NullWritable> {
        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            double sumX = 0;
            double sumY = 0;
            int count = 0;

            // Calculate the sum of all points assigned to this cluster
            for (Text value : values) {
                String[] parts = value.toString().split(",");
                sumX += Double.parseDouble(parts[0]);
                sumY += Double.parseDouble(parts[1]);
                count++;
                context.write(new Text(key.toString() + "," + value.toString()), NullWritable.get());
            }

            // Update the centroid for this cluster
            double centroidX = sumX / count;
            double centroidY = sumY / count;

            String output = "c" + key.toString() + "," + String.valueOf(centroidX) + "," + String.valueOf(centroidY);
            context.write(new Text(output), NullWritable.get());
        }
    }

    public static void writeToFinalOutput(FileSystem fs, int k, String inputPath, String outputCluster,
            String outputClasses) throws IOException, InterruptedException {
        List<List<Point>> points = new ArrayList<>(k);
        // Initialize each element in the list
        for (int i = 0; i < k; i++) {
            points.add(new ArrayList<>());
        }

        List<Point> centroids = new ArrayList<>();

        FileStatus[] files = fs.listStatus(new Path(inputPath));
        for (FileStatus file : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(file.getPath())))) {
                String line;
                while ((line = br.readLine()) != null) {
                    try {
                        String[] parts = line.trim().split(",");
                        int cl;
                        double x = Double.parseDouble(parts[1]);
                        double y = Double.parseDouble(parts[2]);
                        if (parts[0].charAt(0) == 'c') {
                            cl = Integer.parseInt(parts[0].substring(1)) - 1;
                            centroids.add(cl, new Point(x, y));
                        } else {
                            cl = Integer.parseInt(parts[0]) - 1;
                            points.get(cl).add(new Point(x, y));
                        }
                    } catch (Exception e) {
                        continue;
                    }
                }
                br.close();
            }
        }

        // cluster centers
        OutputStream os = fs.create(new Path(outputCluster));

        // Write content to the file
        BufferedWriter br = new BufferedWriter(new OutputStreamWriter(os));
        for (int i = 0; i < k; i++) {
            Point centroid = centroids.get(i);
            br.write((i + 1) + "\t" + centroid.getX() + "," + centroid.getY());
            br.newLine();
        }
        br.close();

        // classes
        os = fs.create(new Path(outputClasses));

        // Write content to the file
        br = new BufferedWriter(new OutputStreamWriter(os));
        for (int i = 0; i < points.size(); i++) {
            for (int j = 0; j < points.get(i).size(); j++) {
                Point point = points.get(i).get(j);
                br.write((i + 1) + "\t" + point.getX() + "," + point.getY());
                br.newLine();
            }
        }
        br.close();
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2 && args.length != 4) {
            System.err.println("Usage: [k maxIterations] inputPath outputPath");
            System.exit(1); // Exit with an error code
        }
        final double THRESHOLD = 0.001;
        int k = 3;
        int maxIterations = 20;
        String inputPathS = args[0];
        String outputPathS = args[1];

        if (args.length == 4) {
            k = Integer.parseInt(args[0]);
            maxIterations = Integer.parseInt(args[1]);
            inputPathS = args[2];
            outputPathS = args[3];
        }

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        List<Point> centroids = initializeCentroids(k, fs, inputPathS);
        saveCentroidsToConf(conf, centroids);

        Path inputPath = new Path(inputPathS); // Initial input path
        Path outputPath;
        String finalPathS = "";
        String subPathS;
        for (int i = 0; i < maxIterations; i++) {
            Job job = Job.getInstance(conf, "K means 2D - Iteration " + (i + 1));

            job.setJarByClass(KMeans2D.class);
            job.setMapperClass(ClusterMapper.class);
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Text.class);

            job.setReducerClass(ClusterReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(NullWritable.class);

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

            List<Point> newCentroids = loadCentroids(k, fs, subPathS);
            // if the number of cluster change (empty cluster appears), stop
            if (newCentroids.size() != k) {
                finalPathS = outputPath.toString();
                break;
            }

            double maxDistance = Double.MIN_VALUE;
            for (int j = 0; j < centroids.size(); j++) {
                double c_distance = calculate_distance(centroids.get(j), newCentroids.get(j));
                if (c_distance > maxDistance) {
                    maxDistance = c_distance;
                }
            }

            if (maxDistance < THRESHOLD || i == maxIterations - 1) {
                finalPathS = outputPath.toString();
                break;
            }

            centroids = newCentroids;
            saveCentroidsToConf(conf, centroids);
        }

        if (finalPathS != "") {
            // if all success
            writeToFinalOutput(
                    fs, k, finalPathS,
                    "./task_2_1.clusters", "./task_2_1.classes");
        }

        fs.close();
        System.exit(0);
    }

}
