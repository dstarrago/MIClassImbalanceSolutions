/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package samplingExperimenter;

import java.util.ArrayList;

/**
 *
 * @author Danel
 */
public class DataCollection {

  private ArrayList<DataEntry> dataEntries;
  //private String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #2/";
  //private String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #2/_BALANCED/BagSmoteKDE/";
  private String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_BOOK DATA/";
  //private String path = "../Data/";

  public DataCollection() {
    dataEntries = new ArrayList<DataEntry>();
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #2/";
    //String path = "../Data/";
    BookCollection();
  }

  private void bioInformatic() {
    Musk();
    mutagenesis();
    AntDrugs();
  }

  private void BookCollection() {
    dataEntries.add(new DataEntry("Musk1", path, "Musk1", 5));
    dataEntries.add(new DataEntry("Musk2", path, "Musk2", 5));
    dataEntries.add(new DataEntry("Atoms_Norm", path, "Atoms_Norm", 5));
    dataEntries.add(new DataEntry("Bonds_Norm", path, "Bonds_Norm", 5));
    dataEntries.add(new DataEntry("Chains_Norm", path, "Chains_Norm", 5));
    dataEntries.add(new DataEntry("WIR-U7", path, "WIR-U7", 5));
    dataEntries.add(new DataEntry("TREC9-C4", path, "TREC9-C4", 5));
    dataEntries.add(new DataEntry("Beach", path, "Beach", 5));
    dataEntries.add(new DataEntry("Fox", path, "Fox", 5));
  }

  private void DummyCollection() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #3/";
    //dataEntries.add(new DataEntry("Dummy1", path, "Dummy1", 1));
    dataEntries.add(new DataEntry("Dummy2", path, "Dummy2", 1));
  }

  private void Musk() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #1/";
    dataEntries.add(new DataEntry("Musk1", path, "01 Musk1", 5));
    dataEntries.add(new DataEntry("Musk2", path, "02 Musk2", 5));
  }

  private void OnlyEastWest() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #1/";
    dataEntries.add(new DataEntry("EastWest", path, "11 EastWest", 5));
  }

  private void OnlySuramin() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #1/";
    dataEntries.add(new DataEntry("Suramin", path, "06 Suramin", 5));
  }

  private void fastTest() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #2/";
    //String path = "../Data/";
    dataEntries.add(new DataEntry("WIRSel-1", path, "26 WIR1", 5));
    dataEntries.add(new DataEntry("WIRSel-2", path, "27 WIR2", 5));
    dataEntries.add(new DataEntry("WIRSel-3", path, "28 WIR3", 5));
    dataEntries.add(new DataEntry("Corel20-1", path, "35 Corel1", 5));
    dataEntries.add(new DataEntry("Corel20-2", path, "36 Corel2", 5));
    dataEntries.add(new DataEntry("Corel20-3", path, "37 Corel3", 5));
  }

  private void compileThioredoxin() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #2/";
    //String path = "../Data/";
    dataEntries.add(new DataEntry("Thioredoxin", path, "07 Thioredoxin", 5));
  }

  private void compileBioCreativeCollection() {
//    String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #2/";
    //String path = "../Data/";
    dataEntries.add(new DataEntry("Process", path, "13 Process", 5));
    dataEntries.add(new DataEntry("Function", path, "14 Function", 5));
    dataEntries.add(new DataEntry("Component", path, "15 Component", 5));
  }

  private void CorelImbalancedCollection() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #2/";
    //String path = "../Data/";
    dataEntries.add(new DataEntry("Corel20-1", path, "35 Corel1", 5));
    dataEntries.add(new DataEntry("Corel20-2", path, "36 Corel2", 5));
    dataEntries.add(new DataEntry("Corel20-3", path, "37 Corel3", 5));
    dataEntries.add(new DataEntry("Corel20-4", path, "38 Corel4", 5));
    dataEntries.add(new DataEntry("Corel20-5", path, "39 Corel5", 5));
    dataEntries.add(new DataEntry("Corel20-6", path, "40 Corel6", 5));
    dataEntries.add(new DataEntry("Corel20-7", path, "41 Corel7", 5));
    dataEntries.add(new DataEntry("Corel20-8", path, "42 Corel8", 5));
    dataEntries.add(new DataEntry("Corel20-9", path, "43 Corel9", 5));
    dataEntries.add(new DataEntry("Corel20-10", path, "44 Corel10", 5));
    dataEntries.add(new DataEntry("Corel20-11", path, "45 Corel11", 5));
    dataEntries.add(new DataEntry("Corel20-12", path, "46 Corel12", 5));
    dataEntries.add(new DataEntry("Corel20-13", path, "47 Corel13", 5));
    dataEntries.add(new DataEntry("Corel20-14", path, "48 Corel14", 5));
    dataEntries.add(new DataEntry("Corel20-15", path, "49 Corel15", 5));
    dataEntries.add(new DataEntry("Corel20-16", path, "50 Corel16", 5));
    dataEntries.add(new DataEntry("Corel20-17", path, "51 Corel17", 5));
    dataEntries.add(new DataEntry("Corel20-18", path, "52 Corel18", 5));
    dataEntries.add(new DataEntry("Corel20-19", path, "53 Corel19", 5));
    dataEntries.add(new DataEntry("Corel20-20", path, "54 Corel20", 5));
  }

  private void WIRImbalancedCollection() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #2/";
    //String path = "../Data/";
    dataEntries.add(new DataEntry("WIRSel-1", path, "26 WIR1", 5));
    dataEntries.add(new DataEntry("WIRSel-2", path, "27 WIR2", 5));
    dataEntries.add(new DataEntry("WIRSel-3", path, "28 WIR3", 5));
    dataEntries.add(new DataEntry("WIRSel-4", path, "29 WIR4", 5, 0));
    dataEntries.add(new DataEntry("WIRSel-5", path, "30 WIR5", 5, 0));
    dataEntries.add(new DataEntry("WIRSel-6", path, "31 WIR6", 5, 0));
  }

  private void animalImbalanced() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #2/";
    //String path = "../Data/";
    dataEntries.add(new DataEntry("Elephant", path, "55 Elephant-Imbal", 5));
    dataEntries.add(new DataEntry("Fox", path, "56 Fox-Imbal", 5));
    dataEntries.add(new DataEntry("Tiger", path, "57 Tiger-Imbal", 5));
  }

  private void mutagenesisImbalanced() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #2/";
    //String path = "../Data/";
    dataEntries.add(new DataEntry("Mutagenesis_atoms", path, "58 Atoms-Imbal", 5));
    dataEntries.add(new DataEntry("Mutagenesis_bonds", path, "59 Bonds-Imbal", 5));
    dataEntries.add(new DataEntry("Mutagenesis_chains", path, "60 Chains-Imbal", 5));
  }

  private void mutagenesis() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #1/";
    //String path = "../Data/";
    dataEntries.add(new DataEntry("Atoms", path, "03 Atoms", 5));
    dataEntries.add(new DataEntry("Bonds", path, "04 Bonds", 5));
    dataEntries.add(new DataEntry("Chains", path, "05 Chains", 5));
  }

  private void AntDrugs() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #1/";
    //String path = "../Data/";
    dataEntries.add(new DataEntry("AntDrugs5", path, "16 AntDrugs5", 5));
    dataEntries.add(new DataEntry("AntDrugs10", path, "17 AntDrugs10", 5));
    dataEntries.add(new DataEntry("AntDrugs20", path, "18 AntDrugs20", 5));
  }

  private void compileCollection1ButSuramin() {
    //String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #1/";
    dataEntries.add(new DataEntry("Musk1", path, "01 Musk1", 5));
    dataEntries.add(new DataEntry("Musk2", path, "02 Musk2", 5));
    dataEntries.add(new DataEntry("Atoms", path, "03 Atoms", 5));
    dataEntries.add(new DataEntry("Bonds", path, "04 Bonds", 5));
    dataEntries.add(new DataEntry("Chains", path, "05 Chains", 5));
    dataEntries.add(new DataEntry("Elephant", path, "08 Elephant", 5));
    dataEntries.add(new DataEntry("Fox", path, "09 Fox", 5));
    dataEntries.add(new DataEntry("Tiger", path, "10 Tiger", 5));
    dataEntries.add(new DataEntry("EastWest", path, "11 EastWest", 5));
    dataEntries.add(new DataEntry("WestEast", path, "12 WestEast", 5));
    dataEntries.add(new DataEntry("AntDrugs5", path, "16 AntDrugs5", 5));
    dataEntries.add(new DataEntry("AntDrugs10", path, "17 AntDrugs10", 5));
    dataEntries.add(new DataEntry("AntDrugs20", path, "18 AntDrugs20", 5));
    dataEntries.add(new DataEntry("TREC9Sel-1", path, "19 TREC9-1", 5));
    dataEntries.add(new DataEntry("TREC9Sel-2", path, "20 TREC9-2", 5));
    dataEntries.add(new DataEntry("TREC9Sel-3", path, "21 TREC9-3", 5));
    dataEntries.add(new DataEntry("TREC9Sel-4", path, "22 TREC9-4", 5));
    dataEntries.add(new DataEntry("TREC9Sel-7", path, "23 TREC9-7", 5));
    dataEntries.add(new DataEntry("TREC9Sel-9", path, "24 TREC9-9", 5));
    dataEntries.add(new DataEntry("TREC9Sel-10", path, "25 TREC9-10", 5));
    dataEntries.add(new DataEntry("WIRSel-7", path, "32 WIR7", 5));
    dataEntries.add(new DataEntry("WIRSel-8", path, "33 WIR8", 5));
    dataEntries.add(new DataEntry("WIRSel-9", path, "34 WIR9", 5));
    dataEntries.add(new DataEntry("CLJ-16.30.2", path, "55 CLJ-16.30.2", 5));
    dataEntries.add(new DataEntry("CLJ-16-50-2", path, "56 CLJ-16-50-2", 5));
    dataEntries.add(new DataEntry("CLJ-80.166.1", path, "57 CLJ-80.166.1", 5));
    dataEntries.add(new DataEntry("CLJ-80.166.1-Strong", path, "58 CLJ-80.166.1-Strong", 5));
    dataEntries.add(new DataEntry("CLJ-80-206-1", path, "59 CLJ-80-206-1", 5));
    dataEntries.add(new DataEntry("CLJ-160.166.1", path, "60 CLJ-160.166.1", 5));
    dataEntries.add(new DataEntry("CLJ-160.166.1-Strong", path, "61 CLJ-160.166.1-Strong", 5));
    dataEntries.add(new DataEntry("CLJ-160-566-1", path, "62 CLJ-160-566-1", 5));
    dataEntries.add(new DataEntry("Corel01vs02", path, "63 Corel01vs02", 5));
    dataEntries.add(new DataEntry("Corel01vs03", path, "64 Corel01vs03", 5));
    dataEntries.add(new DataEntry("Corel01vs04", path, "65 Corel01vs04", 5));
    dataEntries.add(new DataEntry("Corel01vs05", path, "66 Corel01vs05", 5));
    dataEntries.add(new DataEntry("Corel02vs03", path, "67 Corel02vs03", 5));
    dataEntries.add(new DataEntry("Corel02vs04", path, "68 Corel02vs04", 5));
    dataEntries.add(new DataEntry("Corel02vs05", path, "69 Corel02vs05", 5));
    dataEntries.add(new DataEntry("Corel03vs04", path, "70 Corel03vs04", 5));
    dataEntries.add(new DataEntry("Corel03vs05", path, "71 Corel03vs05", 5));
    dataEntries.add(new DataEntry("Corel04vs05", path, "72 Corel04vs05", 5));
  }

  public int numDatasets() {
    return dataEntries.size();
  }

  public DataEntry dataEntry(int index) {
    return dataEntries.get(index);
  }

}
