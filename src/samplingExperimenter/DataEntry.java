/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package samplingExperimenter;

/**
 *
 * @author Danel
 */
public class DataEntry {

  public static String VALIDATION_MASK = "%dCV";

  private String name;
  private String path;
  private String folder;
  private int numFolds;
  private int numRuns;
  private int posClassLabel = 1;
  private String validationDirMask;
  private String fileNameMask;

  public DataEntry(String name, String path, String folder, int numRuns, int numFolds,
          int posClassLabel, String validationDirMask, String fileNameMask) {
    this.name = name;
    this.path = path;
    if (!path.endsWith("/")) this.path += "/";
    this.folder = folder;
    if (!folder.endsWith("/")) this.folder += "/";
    this.numFolds = numFolds;
    this.numRuns = numRuns;
    this.posClassLabel = posClassLabel;
    this.validationDirMask = validationDirMask;
    if (!validationDirMask.endsWith("/")) this.validationDirMask += "/";
    this.fileNameMask = fileNameMask;
  }

  public DataEntry(String name, String path, String folder, int numFolds,
          int posClassLabel, String validationDirMask, String fileNameMask) {
    this(name, path, folder, 1, numFolds, posClassLabel, "%d-Folds-CV/", "%s-f%d-%s.arff");
  }

  public DataEntry(String name, String path, String folder, int numRuns, int numFolds,
          int posClassLabel) {
    this(name, path, folder, numRuns, numFolds, posClassLabel, "%d-Times-%d-Folds-CV/", "%s-r%d-f%d-%s.arff");
  }

  public DataEntry(String name, String path, String folder, int numFolds,
          int posClassLabel) {
    this(name, path, folder, numFolds, posClassLabel, "%d-Folds-CV/", "%s-f%d-%s.arff");
  }

  public DataEntry(String name, String path, String folder, int numFolds) {
    this(name, path, folder, numFolds, 1, "%d-Folds-CV/", "%s-f%d-%s.arff");
  }

  public String name() {
    return name;
  }

  public String path() {
    return path;
  }

  public String folder() {
    return folder;
  }

  public int numFolds() {
    return numFolds;
  }

  public int numRuns() {
    return numRuns;
  }

  public int posClassLabel() {
    return posClassLabel;
  }

  public String validationDirMask() {
    return validationDirMask;
  }

  public String fileNameMask() {
    return fileNameMask;
  }

  public String fold(String stage, int run, int fold) {
    String s = null;
    if (numRuns > 1) {
      s = path + folder + String.format(validationDirMask, numRuns, numFolds) +
          String.format(fileNameMask, name, run, fold, stage);
    } else {
      s = path + folder + String.format(validationDirMask, numFolds) +
          String.format(fileNameMask, name, fold, stage);
    }
    return s;
  }

  public String fold(String stage, int fold) {
   String s = path + folder + String.format(validationDirMask, numFolds) +
          String.format(fileNameMask, name, fold, stage);
    return s;
  }
  
  public String foldX(String branch, String stage, int run, int fold) {
    String s = null;
    if (numRuns > 1) {
      s = path + branch + "/"+ folder + String.format(validationDirMask, numRuns, numFolds) +
          String.format(fileNameMask, name, run, fold, stage);
    } else {
      s = path + branch + "/"+ folder + String.format(validationDirMask, numFolds) +
          String.format(fileNameMask, name, fold, stage);
    }
    return s;
  }

  public String foldX(String branch, String stage, int fold) {
    String s = path + branch + "/"+ folder + String.format(validationDirMask, numFolds) +
            String.format(fileNameMask, name, fold, stage);
    return s;
  }

  public String fileName() {
    return path + folder + name + ".arff";
  }

  public String trainFold(int fold) {
    return fold("train", fold);
  }

  public String testFold(int fold) {
    return fold("test", fold);
  }

  public String balTrainFold(int fold) {
    return fold("balancedTrain", fold);
  }

  public String trainFold(int run, int fold) {
    return fold("train", run, fold);
  }

  public String testFold(int run, int fold) {
    return fold("test", run, fold);
  }

  public String balTrainFold(int run, int fold) {
    return fold("balancedTrain", run, fold);
  }

}
