/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package samplingExperimenter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import jxl.read.biff.BiffException;
import weka.classifiers.Evaluation;

import jxl.Workbook;
import jxl.format.Alignment;
import jxl.format.Colour;
import jxl.write.Label;
import jxl.write.Number;
import jxl.write.NumberFormat;
import jxl.write.WritableSheet;
import jxl.write.WritableWorkbook;
import jxl.write.WriteException;
import jxl.write.WritableCellFormat;

/**
 *
 * @author Danel
 */
public class ResultsToExcel {

  /**
   * Data ordering in the excel sheet
   */
  private final int DATASET         = 0;
  private final int AUC             = 1;
  private final int GMEAN           = 2;
  private final int F1              = 3;
  private final int PRECISION       = 4;
  private final int RECALL          = 5;
  private final int KAPPA           = 6;
  private final int ACCURACY        = 7;
  private final int TP              = 8;
  private final int FP              = 9;
  private final int FN              = 10;
  private final int TN              = 11;
  private final int TRAIN_TIME      = 12;
  private final int TEST_TIME       = 13;
  private final int POSITIVE_CLASS  = 14;
  private final int VALIDATION      = 15;

  private final String[] columnNames =
                    {"Dataset",
                     "AUC",
                     "GMean",
                     "F1",
                     "Precision",
                     "Recall",
                     "Kappa",
                     "Accuracy",
                     "TP",
                     "FP",
                     "FN",
                     "TN",
                     "Train time (secs)",
                     "Test time (secs)",
                     "Pos class",
                     "Validation"};

  private final int defaultColumnWidth = 20;
  private final int initialSheetsCapacity = 10;
  private final int headerHeight = 2;
  private String name;
  private File bookFile;
  private WritableWorkbook resultsBook;
  private ArrayList<WritableSheet> sheets;
  private ArrayList<Integer> recordCount;
  private int selectedSheet;
  private String outputDir;
  private WritableCellFormat datasetsFormat;
  private WritableCellFormat centerFormat;
  private WritableCellFormat headerFormat;
  private WritableCellFormat dp4cell;
  private WritableCellFormat dp2cell;

  public ResultsToExcel(String name, String outputDir) {
    this.name = name;
    this.outputDir = outputDir;
    sheets = new ArrayList<WritableSheet>(initialSheetsCapacity);
    recordCount = new ArrayList<Integer>(initialSheetsCapacity);
  }

  public void openBook() throws IOException, WriteException {
    bookFile = new File(outputDir, name + ".xls");
    resultsBook = Workbook.createWorkbook(bookFile);

    centerFormat = new WritableCellFormat();
    centerFormat.setAlignment(Alignment.CENTRE);

    headerFormat = new WritableCellFormat();
    headerFormat.setAlignment(Alignment.CENTRE);
    headerFormat.setBackground(Colour.TAN);

    datasetsFormat = new WritableCellFormat();
    datasetsFormat.setAlignment(Alignment.LEFT);

    NumberFormat dp4 = new NumberFormat("0.0000");
    dp4cell = new WritableCellFormat(dp4);
    dp4cell.setAlignment(Alignment.CENTRE);

    NumberFormat dp2 = new NumberFormat("0.00");
    dp2cell = new WritableCellFormat(dp2);
    dp2cell.setAlignment(Alignment.CENTRE);
    
  }

  public void closeBook() throws IOException, WriteException {
    resultsBook.write();
    resultsBook.close();
  }

  public void addNewSheet(String name, String comments) throws WriteException {
    sheets.add(resultsBook.createSheet(name, sheets.size()));
    selectSheet(sheets.size() - 1);
    writeHeader(selectedSheet, comments);
    recordCount.add(0);
  }

  public void addNewSheet(String name) throws WriteException {
    addNewSheet(name, null);
  }
  
  public void selectSheet(int index) {
    selectedSheet = index;
  }

  private void writeHeader(int sheetIndex, String comments) throws WriteException {
    if (comments != null) {
      Label comm = new Label(0, 0, comments);
      sheets.get(sheetIndex).addCell(comm);
    }
    for (int i = 0; i < columnNames.length; i++) {
      sheets.get(sheetIndex).setColumnView(i, defaultColumnWidth);
      Label label = new Label(i, 1, columnNames[i], headerFormat);
      sheets.get(sheetIndex).addCell(label);
    }
  }

  public void addResult(DataEntry data, Evaluation eval,
          double trainTime, double testTime) throws WriteException, IOException {

    Label label;
    Number number;

    label= new Label(DATASET, headerHeight + recordCount.get(selectedSheet),
            data.name(), datasetsFormat);
    sheets.get(selectedSheet).addCell(label);

    if (data.numRuns() > 1) {
      label= new Label(VALIDATION, headerHeight + recordCount.get(selectedSheet),
              String.format(data.validationDirMask(), data.numRuns(), data.numFolds()), centerFormat);
    } else {
      label= new Label(VALIDATION, headerHeight + recordCount.get(selectedSheet),
              String.format(data.validationDirMask(), data.numFolds()), centerFormat);
    }
    sheets.get(selectedSheet).addCell(label);

    number = new Number(POSITIVE_CLASS, headerHeight + recordCount.get(selectedSheet),
            data.posClassLabel(), centerFormat);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(TP, headerHeight + recordCount.get(selectedSheet),
            eval.numTruePositives(data.posClassLabel()), centerFormat);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(FP, headerHeight + recordCount.get(selectedSheet),
            eval.numFalsePositives(data.posClassLabel()), centerFormat);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(FN, headerHeight + recordCount.get(selectedSheet),
            eval.numFalseNegatives(data.posClassLabel()), centerFormat);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(TN, headerHeight + recordCount.get(selectedSheet),
            eval.numTrueNegatives(data.posClassLabel()), centerFormat);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(ACCURACY, headerHeight + recordCount.get(selectedSheet),
            eval.pctCorrect(), dp2cell);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(KAPPA, headerHeight + recordCount.get(selectedSheet),
            eval.kappa(), dp4cell);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(AUC, headerHeight + recordCount.get(selectedSheet),
            eval.areaUnderROC(data.posClassLabel()), dp4cell);
    sheets.get(selectedSheet).addCell(number);

    double gmean = Math.sqrt( eval.truePositiveRate(data.posClassLabel()) *
                              eval.trueNegativeRate(data.posClassLabel()));

    number = new Number(GMEAN, headerHeight + recordCount.get(selectedSheet),
            gmean, dp4cell);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(RECALL, headerHeight + recordCount.get(selectedSheet),
            eval.recall(data.posClassLabel()), dp4cell);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(PRECISION, headerHeight + recordCount.get(selectedSheet),
            eval.precision(data.posClassLabel()), dp4cell);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(F1, headerHeight + recordCount.get(selectedSheet),
            eval.fMeasure(data.posClassLabel()), dp4cell);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(TRAIN_TIME, headerHeight + recordCount.get(selectedSheet),
            trainTime, dp2cell);
    sheets.get(selectedSheet).addCell(number);

    number = new Number(TEST_TIME, headerHeight + recordCount.get(selectedSheet),
            testTime, dp2cell);
    sheets.get(selectedSheet).addCell(number);

    recordCount.set(selectedSheet, recordCount.get(selectedSheet) + 1);
  }

  public String name() {
    return name;
  }
  
  public String outputDir() {
    return outputDir;
  }

}
