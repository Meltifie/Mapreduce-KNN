import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import king.Utils.Distance;
import king.Utils.ListWritable;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

/**
 * KNearestNeigbour Classifier
 * each instance in training set is of form a1,a2,a3...an,l1
 * in which l1 represents the label. and each instance in
 * predict set is of form a1,a2,a3...an,-1,in which -1 is the
 * label we want to specify.
 * In my algorithm,I assume that the trainning set is relatively
 * small so we can load them in memory and the predict set is large
 * another thing we need to pay attention to is that all our test
 * instances are all in one file so that the index of line is unique
 * to each instance.
 * @author KING
 *
 */
public class KNearestNeighbour {
    public static class KNNMap extends Mapper<LongWritable,
            Text,LongWritable,ListWritable<DoubleWritable>>{
        private int k;
        private ArrayList<Instance> trainSet;
        private Configuration conf;

        @Override
        protected void setup(Context context) throws IOException,InterruptedException{
            conf = context.getConfiguration();
            k = conf.getInt("k", 1);
            trainSet = new ArrayList<Instance>();

            URI[] trainFile = Job.getInstance(conf).getCacheFiles();
            //add all the training instances into attributes
            BufferedReader br = null;
            String line;
            for(int i = 0;i < trainFile.length;i++){
                br = new BufferedReader(new FileReader(trainFile[0].toString()));
                while((line = br.readLine()) != null){
                    Instance trainInstance = new Instance(line);
                    trainSet.add(trainInstance);
                }
            }
        }

        /**
         * find the nearest k labels and put them in an object
         * of type ListWritable. and emit <textIndex,LabelList>
         */
        @Override
        public void map(LongWritable textIndex, Text textLine, Context context)
                throws IOException, InterruptedException {
            //distance stores all the current nearest distance value.
            //trainLabel store the corresponding Label
            ArrayList<Double> distance = new ArrayList<Double>(k);
            ArrayList<DoubleWritable> trainLabel = new ArrayList<DoubleWritable>(k);
            for(int i = 0;i < k;i++){
                distance.add(Double.MAX_VALUE);
                trainLabel.add(new DoubleWritable(-1.0));
            }
            ListWritable<DoubleWritable> Labels = new ListWritable<DoubleWritable>(DoubleWritable.class);
            Instance testInstance = new Instance(textLine.toString());
            for(int i = 0;i < trainSet.size();i++){
                try {
                    double dis = Distance.EuclideanDistance(trainSet.get(i).getAtrributeValue(), testInstance.getAtrributeValue());
                    int index = indexOfMax(distance);
                    if(dis < distance.get(index)){
                        distance.remove(index);
                        trainLabel.remove(index);
                        distance.add(dis);
                        trainLabel.add(new DoubleWritable(trainSet.get(i).getLabel()));
                    }
                } catch (Exception e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
            Labels.setList(trainLabel);
            context.write(textIndex, Labels);
        }

        /**
         * return the index of the maximum number of an array
         * @param array
         * @return
         */
        public int indexOfMax(ArrayList<Double> array){
            int index = -1;
            Double min = Double.MIN_VALUE;
            for (int i = 0;i < array.size();i++){
                if(array.get(i) > min){
                    min = array.get(i);
                    index = i;
                }
            }
            return index;
        }
    }

    public static class KNNReduce extends Reducer<LongWritable,ListWritable<DoubleWritable>,NullWritable,DoubleWritable>{

        @Override
        public void reduce(LongWritable index, Iterable<ListWritable<DoubleWritable>> kLabels, Context context)
                throws IOException, InterruptedException{
            /**
             * each index can actually have one list because of the
             * assumption that the particular line index is unique
             * to one instance.
             */
            DoubleWritable predictedLabel = new DoubleWritable();
            for(ListWritable<DoubleWritable> val: kLabels){
                try {
                    predictedLabel = valueOfMostFrequent(val);
                    break;
                } catch (Exception e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
            context.write(NullWritable.get(), predictedLabel);
        }

        public DoubleWritable valueOfMostFrequent(ListWritable<DoubleWritable> list) throws Exception{
            if(list.isEmpty())
                throw new Exception("list is empty!");
            else{
                HashMap<DoubleWritable,Integer> tmp = new HashMap<DoubleWritable,Integer>();
                for(int i = 0 ;i < list.size();i++){
                    if(tmp.containsKey(list.get(i))){
                        Integer frequence = tmp.get(list.get(i)) + 1;
                        tmp.remove(list.get(i));
                        tmp.put(list.get(i), frequence);
                    }else{
                        tmp.put(list.get(i), new Integer(1));
                    }
                }
                //find the value with the maximum frequence.
                DoubleWritable value = new DoubleWritable();
                Integer frequence = new Integer(Integer.MIN_VALUE);
                Iterator<Entry<DoubleWritable, Integer>> iter = tmp.entrySet().iterator();
                while (iter.hasNext()) {
                    Map.Entry<DoubleWritable,Integer> entry = (Map.Entry<DoubleWritable,Integer>) iter.next();
                    if(entry.getValue() > frequence){
                        frequence = entry.getValue();
                        value = entry.getKey();
                    }
                }
                return value;
            }
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException{
        Configuration conf = new Configuration();
        Job kNNJob = Job.getInstance(conf, "kNNJob");
        kNNJob.setJarByClass(KNearestNeighbour.class);
        kNNJob.addCacheFile(URI.create(args[2]));
        kNNJob.getConfiguration().setInt("k", Integer.parseInt(args[3]));

        kNNJob.setMapperClass(KNNMap.class);
        kNNJob.setMapOutputKeyClass(LongWritable.class);
        kNNJob.setMapOutputValueClass(ListWritable.class);

        kNNJob.setReducerClass(KNNReduce.class);
        kNNJob.setOutputKeyClass(NullWritable.class);
        kNNJob.setOutputValueClass(DoubleWritable.class);

        kNNJob.setInputFormatClass(TextInputFormat.class);
        kNNJob.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.addInputPath(kNNJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(kNNJob, new Path(args[1]));

        kNNJob.waitForCompletion(true);
        System.out.println("finished!");
    }
}
