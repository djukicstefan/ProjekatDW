import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.attribute.Attribute;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

import scala.Tuple2;
public class Main {
    public static String[] classes;

    public static InputStreamReader getReader(String path){
        try {
            return new InputStreamReader(new FileInputStream(path), "UTF-8");
        } catch(Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static JavaRDD<LabeledPoint> loadData(SparkSession session, String path){

        List<Row> data = new ArrayList<Row>();

        CsvParserSettings settings= new CsvParserSettings();
        settings.getFormat().setLineSeparator("\n");
        CsvParser parser = new CsvParser(settings);
        parser.beginParsing(getReader(path));

        String[] row = parser.parseNext();
        while((row = parser.parseNext()) != null){
            try
            {
                int id = Integer.parseInt(row[0].trim());
                String strClass = row[row.length-1].trim();
                Double[] features = new Double[row.length-2];
                for(int i = 1; i < row.length-1; i++)
                    features[i-1] = Double.parseDouble(row[i].trim());
                data.add(RowFactory.create(id,features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9],strClass));
            }
            catch(Exception ex) {
                continue;
            }

        }

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("Area", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("MajorAxisLength", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("MinorAxisLength", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("Eccentricity", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("ConvexArea", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("EquivDiameter", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("Extent", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("Perimeter", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("Roundness", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("AspectRation", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("Class", DataTypes.StringType, false, Metadata.empty()),
        });

        Dataset<Row> dataFrame = session.sqlContext().createDataFrame(data, schema);
        dataFrame = dataFrame.drop("id");

        StringIndexer indexer = new StringIndexer().setInputCol("Class").setOutputCol("ClassIndex");
        StringIndexerModel indexerModel = indexer.fit(dataFrame);
        dataFrame = indexerModel.transform(dataFrame);
        StructField inputColSchema = dataFrame.schema().apply(indexer.getOutputCol());
        classes = Attribute.fromStructField(inputColSchema).toMetadata().getMetadata("ml_attr").getStringArray("vals");

        VectorAssembler assembler = new VectorAssembler().setInputCols(new String[]{"Area", "MajorAxisLength", "MinorAxisLength","Eccentricity","ConvexArea","EquivDiameter","Extent", "Perimeter", "Roundness","AspectRation"}).setOutputCol("features");
        dataFrame = assembler.transform(dataFrame);
        dataFrame = dataFrame.drop("Area").drop("MajorAxisLength").drop("MinorAxisLength").drop("Eccentricity").drop("ConvexArea").drop("EquivDiameter").drop("Extent").drop("Perimeter").drop("Roundness").drop("AspectRation");

        MinMaxScaler scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures");
        dataFrame= scaler.fit(dataFrame).transform(dataFrame);



        JavaSparkContext jc = JavaSparkContext.fromSparkContext(session.sparkContext());
        dataFrame.show(3,false);
        //dataFrame.printSchema();

        //1,3
        List<Row> rows = dataFrame.collectAsList();
        List<LabeledPoint> lbps = new ArrayList<LabeledPoint>();
        for(Row r :rows) {
            lbps.add(new LabeledPoint(r.getDouble(1),org.apache.spark.mllib.linalg.Vectors.fromML((Vector)r.get(3))));
        }
        return jc.parallelize(lbps);
    }

    private static void writePredictions(JavaRDD<Object> preds) {
        try {
            FileWriter fw = new FileWriter("predictions.csv");

            List<Object> predicted=preds.collect();

            for(int i=0;i<predicted.size();i++){
                fw.write(classes[new Double(predicted.get(i).toString()).intValue()]+ "\n");
            }
            fw.close();
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private static NaiveBayesModel trainNaiveBayes(JavaRDD<LabeledPoint> trainingSet) {
        return NaiveBayes.train(trainingSet.rdd(), 1.0);
    }

    private static LogisticRegressionModel trainLogisticRegression(JavaRDD<LabeledPoint> trainingSet) {
        return new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingSet.rdd());
    }

    private static SVMModel trainSVM(JavaRDD<LabeledPoint> trainingSet) {
        return  SVMWithSGD.train(trainingSet.rdd(), 100);
    }

    private static DecisionTreeModel trainDecisionTree(JavaRDD<LabeledPoint> trainingSet) {

        int numClasses = 2;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String impurity = "gini";
        int maxDepth = 5;
        int maxBins = 32;

        // Train a DecisionTree model for classification.
        return DecisionTree.trainClassifier(trainingSet, numClasses,categoricalFeaturesInfo, impurity, maxDepth, maxBins);
    }

    private static void writePredictionsWithExpected(JavaPairRDD<Object, Object> preds) {

        try {
            FileWriter fw = new FileWriter("predictions.csv");
            List<Double> predicted=new ArrayList<Double>();
            List<Double> expected=new ArrayList<Double>();

            preds.take(Integer.parseInt(Long.toString(preds.count()))).forEach(tup->{
                predicted.add((Double) tup._1);
                expected.add((Double) tup._2);
            });

            System.out.println(predicted.size());
            for(int i=0;i<predicted.size();i++){
                fw.write(predicted.get(i)+","+expected.get(i) + "\n");
            }
            fw.close();
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void main(String[] args) {
        try {
            FileUtils.forceDelete(new File("D:/spark-models/"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        SparkSession session = SparkSession.builder().appName("RiceClassificator").master("local").getOrCreate();
        SparkContext context = session.sparkContext();

        String path = "D:\\elfak\\IV godina\\Skladištenje podataka i otkrivanje znanja\\ProjekatDW\\Projekat 3\\RiceSeed\\src\\Rice.csv";
        JavaRDD<LabeledPoint> trainingData = loadData(session, path);

        JavaRDD<LabeledPoint>[] tmp = trainingData.randomSplit(new double[]{0.8, 0.2},13156123);
        JavaRDD<LabeledPoint> trainingSet = tmp[0];
        JavaRDD<LabeledPoint> testSet = tmp[1];

        //LogisticRegressionModel model =t rainLogisticRegression(trainingSet);
        NaiveBayesModel model = trainNaiveBayes(trainingSet);
        //SVMModel model = trainSVM(trainingSet);
        //DecisionTreeModel model = trainDecisionTree(trainingSet);


        model.save(context, "D:/spark-models/RiceModel");

        JavaPairRDD<Object, Object> predictionAndLabel = testSet.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        writePredictions(predictionAndLabel.map(p->p._1));

        BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionAndLabel.rdd());
        MulticlassMetrics mMetrics = new MulticlassMetrics(predictionAndLabel.rdd());

        List<String> info = new ArrayList<String>();
        info.add("Accuracy: " + predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) testSet.count());
        info.add("Area under ROC " + metrics.areaUnderROC() + "");
        info.add("Matrix: \n" + mMetrics.confusionMatrix());

        session.close();

        for(String str : info)
            System.out.println(str);

        System.out.print("0-"+classes[0]);
        System.out.print("1-"+classes[1]);
    }
}
