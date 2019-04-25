/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package samplingExperimenter;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Danel
 */
public class Test {

  public void loadAndPrintName() {
    String path = "../Data/";
    String name = "Dummy1.arff";
    String fileName = path + name;
    try {
      Instances data = DataSource.read(fileName);
      System.out.println("Hello machine learning: ");
      System.out.println(data.relationName());
    } catch (Exception e) {
        System.out.println("The file is not there and you know it!");
    }
  }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
      Test L = new Test();
      L.loadAndPrintName();
    }

}
