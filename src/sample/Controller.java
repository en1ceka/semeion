package sample;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Color;
import javafx.scene.text.Text;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;


import javax.imageio.ImageIO;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Random;

public class Controller {

    MultilayerPerceptron mlp;
    double startX, startY, lastX,lastY,oldX,oldY;
    private GraphicsContext gcF, gcG;
    Instances data_struct;
    Image img;

    @FXML
    private Button train_btn;

    @FXML
    private AnchorPane training_info;

    @FXML
    private TextField learning_rate_textfield;

    @FXML
    private TextField momentum_textfield;

    @FXML
    private TextField epochs_textfield;

    @FXML
    private TextField layers_textfield;

    @FXML
    private TextField brush_size;

    @FXML
    private TextField model_name_textfield;

    @FXML
    private ComboBox<?> model_combobox;

    @FXML
    private Canvas canvas;

    @FXML
    private Label evaluate_info;

    @FXML
    private AnchorPane anchor_pane;
    @FXML
    private ComboBox<String> test_data_combobox;



    @FXML
    public void initialize() {
        gcF = canvas.getGraphicsContext2D();
        gcF.setFill(Color.WHITE);
        gcF.fillRect(0,0,160,160);


        test_data_combobox.getItems().add("0%");
        test_data_combobox.getItems().add("10%");
        test_data_combobox.getItems().add("20%");
        test_data_combobox.getItems().add("33%");

        test_data_combobox.getSelectionModel().select(0);

        try{
            String filepath = "./src/data/data_structure.arff";
            FileReader trainreader = new FileReader(filepath);
            data_struct = new Instances(trainreader);
            data_struct.setClassIndex(data_struct.numAttributes() -1);


        }catch (Exception e){
            e.printStackTrace();
        }

        try{
            mlp = (MultilayerPerceptron) weka.core.SerializationHelper.read("./src/models/ggg.model");
        }catch (Exception a) {

        }
    }
    @FXML
    private void onMousePressedListener(MouseEvent e){
        this.startX = e.getX();
        this.startY = e.getY();
        this.oldX = e.getX();
        this.oldY = e.getY();
        System.out.println(startX + " "+  startY);
    }

    @FXML
    private void onMouseDraggedListener(MouseEvent e){
        this.lastX = e.getX();
        this.lastY = e.getY();
        freeDrawing();
    }
    private void pr(Object obj){
        System.out.println(obj);
    }

    @FXML
    private void onMouseReleaseListener(MouseEvent e){
//        pr(mlp.classifyInstance(data_struct.instance(0)));

        File file = new File("./src/data/digit.png");
        WritableImage wim = new WritableImage(160, 160);
        canvas.snapshot(null, wim);
        try{
            ImageIO.write(SwingFXUtils.fromFXImage(wim, null),
                    "png", file);
        }catch (Exception a){}

        BufferedImage img = null;
        try {
            img = ImageIO.read(new File("./src/data/digit.png"));
        } catch (IOException a) {
        }
        BufferedImage after = new BufferedImage(16, 16, BufferedImage.TYPE_INT_ARGB);
        AffineTransform at = new AffineTransform();
        at.scale(0.1, 0.1);
        AffineTransformOp scaleOp =
                new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
        after = scaleOp.filter(img, after);
        File outputfile = new File("./src/data/scaled_digit.png");
        try{
            ImageIO.write(after, "png", outputfile);
        }catch (Exception a){

        }





        double[] instanceValue1 = new double[data_struct.numAttributes()];
        int i = 0;
        for(int y=0;y<16;y++){
            for(int x=0;x<16;x++) {
               if (new java.awt.Color(after.getRGB(x, y)).getRed() < 127) {
                   instanceValue1[i] = 1;
               }else instanceValue1[i] = 0;
               i++;
            }
        }
        instanceValue1[256] = 9;

        data_struct.add(new DenseInstance(1.0, instanceValue1));
        try {
            String val = String.valueOf(mlp.classifyInstance(data_struct.instance(0)));
            evaluate_info.setText(val);
            pr(val + " " + data_struct.numInstances());
            data_struct.remove(0);
        }catch (Exception a){
            a.printStackTrace();
        }
    }



    @FXML
    private void onMouseExitedListener(MouseEvent event)
    {

    }

    private void freeDrawing()
    {
        gcF.setLineWidth(Integer.valueOf(brush_size.getText()));
        gcF.setStroke(Color.BLACK);
        gcF.strokeLine(oldX, oldY, lastX, lastY);
        oldX = lastX;
        oldY = lastY;

    }

    @FXML
    void train_btn_pressed(MouseEvent event) {
//        training_info.setText("TRAINING...");
    }

    @FXML
    void clear_btn_clicked(MouseEvent event) {
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setFill(Color.WHITE);
        gc.fillRect(0,0,160,160);
    }

    Instances filter_inst(Instances data, boolean a, int val){
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();


        // set options for creating the subset of data
        String[] options = new String[6];

        options[0] = "-N";                 // indicate we want to set the number of folds
        options[1] = Integer.toString(val);  // split the data into five random folds
        options[2] = "-F";                 // indicate we want to select a specific fold
        options[3] = Integer.toString(1);  // select the first fold
        options[4] = "-S";                 // indicate we want to set the random seed
        options[5] = Integer.toString(1);  // set the random seed to 1
        Instances dt;
        try{
            if(!a) {
                filter.setOptions(options);        // set the filter options
                filter.setInputFormat(data);       // prepare the filter for the data format
                dt = Filter.useFilter(data, filter);
            }else {
                filter.setOptions(options);        // set the filter options
                filter.setInputFormat(data);       // prepare the filter for the data format
                filter.setInvertSelection(true);
                dt = Filter.useFilter(data, filter);
            }
            return dt;
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }

    }

    void eval_model(MultilayerPerceptron mlp, Instances train, Instances test){
        try{
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(mlp, train);
            System.out.println(eval.toSummaryString());
            double roundOff = Math.round(eval.rootMeanSquaredError() * 10000.0) / 10000.0;



        Text txt = new Text( "\n"
                + " INFO ON TRAIN DATA:" +"\n"
                + " Accuracy: " + Math.round(eval.correct()/eval.numInstances()*10000)/100  + "%\n"
                + " Correctly Classified Instances: " + eval.correct() + "\n"
                + " Incorrectly Classified Instances: " + eval.incorrect() + "\n"
                + " Root mean squared error: " + roundOff + "\n"
                + " Total Number of Instances: " + eval.numInstances() + "\n"
        );

        if(test != null) {
            eval = new Evaluation(train);
            eval.evaluateModel(mlp,test);
            txt.setText(txt.getText() + "\n"
                    + " INFO ON TEST DATA:" +"\n"
                    + " Accuracy: " + Math.round(eval.correct()/eval.numInstances()*10000)/100  + "%\n"
                    + " Correctly Classified Instances: " + eval.correct() + "\n"
                    + " Incorrectly Classified Instances: " + eval.incorrect() + "\n"
                    + " Root mean squared error: " + roundOff + "\n"
                    + " Total Number of Instances: " + eval.numInstances() + "\n");
        }


        txt.wrappingWidthProperty().bind(training_info.widthProperty());
        training_info.getChildren().add(txt);
        }catch (Exception e){
            e.printStackTrace();
        }
    }


    @FXML
    void train_btn_clicked(MouseEvent event) {
        if(!training_info.getChildren().isEmpty()) {
            training_info.getChildren().remove(0);
        }


        try{
            String filepath = "./src/data/semeion.arff";
            FileReader trainreader = new FileReader(filepath);
            Instances data = new Instances(trainreader);
            data.setClassIndex(data.numAttributes() -1);
            Random rand = new Random();
            data.randomize(rand);

            Instances train = data;
            Instances test = null;

            if(test_data_combobox.getSelectionModel().isSelected(1)){
                train = filter_inst(data, true, 10);
                test = filter_inst(data, false, 10);
            }
            if(test_data_combobox.getSelectionModel().isSelected(2)){
                train = filter_inst(data, true, 5);
                test = filter_inst(data, false, 5);
            }
            if(test_data_combobox.getSelectionModel().isSelected(3)){
                train = filter_inst(data, true, 3);
                test = filter_inst(data, false, 3);
            }


            //Instance of NN
            mlp = new MultilayerPerceptron();
            //Setting Parameters
            mlp.setLearningRate(Double.valueOf(learning_rate_textfield.getText()));
            mlp.setMomentum(Double.valueOf(momentum_textfield.getText()));
            mlp.setTrainingTime(Integer.valueOf(epochs_textfield.getText()));
            mlp.setHiddenLayers(layers_textfield.getText());
            mlp.buildClassifier(train);

            eval_model(mlp,train, test);

            weka.core.SerializationHelper.write("./src/models/"+model_name_textfield.getText()+".model", mlp);



        }
        catch(Exception ex){
            ex.printStackTrace();
        }
    }


    private static void appendToFile(Path path, String content)
            throws IOException {

        Files.writeString(path,
                content,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND);
        System.out.println("done");

    }


//    BufferedImage test_img = new BufferedImage(16, 16, BufferedImage.TYPE_INT_ARGB);
//    double[] db;
//            for (int s = 0; s < 1500; s++){
//        pr(s);
//        test_img = new BufferedImage(16, 16, BufferedImage.TYPE_INT_ARGB);
//        db = train.get(s).toDoubleArray();
//
//        int i = 0;
//        for(int y=0;y<16;y++){
//            for(int x=0;x<16;x++) {
//                if (db[i]==1){
//                    test_img.setRGB(x,y, new java.awt.Color(0,0,0).getRGB());
//                }
//                i++;
//            }
//        }
//        File outputfile = new File("./src/data/test_images/" + s + ".png");
//        try{
//            ImageIO.write(test_img, "png", outputfile);
//        }catch (Exception a){
//
//        }
//    }

//    public static BufferedImage toBufferedImage(Image img)
//    {
//        if (img instanceof BufferedImage)
//        {
//            return (BufferedImage) img;
//        }
//
//        // Create a buffered image with transparency
//        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);
//
//        // Draw the image on to the buffered image
//        Graphics2D bGr = bimage.createGraphics();
//        bGr.drawImage(img, 0, 0, null);
//        bGr.dispose();
//
//        // Return the buffered image
//        return bimage;
//    }

}
