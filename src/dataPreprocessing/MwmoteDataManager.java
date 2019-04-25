/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package dataPreprocessing;

import Utils.AveHausdorffDist;
import Utils.BagLevelDistance;
import Utils.CosineDist;
import Utils.DiverseDensity;
import Utils.InstanceLevelDistance;
//import Utils.NEuclideanDist;
import java.util.ArrayList;
//import java.math.BigDecimal;
import java.util.Enumeration;
import weka.core.*;

/**
 * Class that calculate and store data relative to the training dataset. Data are
 * subsequently used for classification and data processing algorithms.
 *
 * @author Danel
 */
public class MwmoteDataManager {

  private static final int defaultNumNNPosFiltering = 5;   // K1
  private static final int defaultNumNNBorderMaj = 3;      // K2
  private static final int defaultNumNNAttrVoting = 5;     // K4
  private static final int needToBeCalculated = -1;

  /**
   * The training dataset
   */
  private Instances data;

  /**
   * Index of the positive class label
   */
  private int posClassLabel;

  /**
   * List of positive example indexes in the training set.
   */
  private ArrayList<Integer> positives;

  /**
   * List of negative example indexes in the training set.
   */
  private ArrayList<Integer> negatives;

  /**
   * List of filtered positive example indexes in the list of positive examples.
   * Those positives examples after removing outliers.
   */
  private ArrayList<Integer> posFiltered;

  /**
   * Positive-to-Positive distance matrix.
   * Is an N x N matrix where N is the number of positive examples.
   * The element (i,j) of the matrix has the distance between the ith example and
   * the jth example in the list of positive examples.
   */
  private double[][] posPosDistances;

  /**
   * Positive-to-Negative distance matrix.
   * Is an N x M matrix where N is the number of positive examples and M is the
   * number of negative examples.
   * The element (i,j) of the matrix has the distance between the ith positive
   * example and the jth negative example in the lists of positive and negatives
   * examples respectively.
   */
  private double[][] posNegDistances;

  /**
   * List of informative positive examples indexes in the list of positive examples.
   */
  private ArrayList<Integer> inforMin;

  /**
   * List of borderline negative examples indexes in the list of negative examples.
   */
  private ArrayList<Integer> borderMaj;

  /**
   * The selection probability of each informative positive example.
   */
  private double[] selectionProb;

  /**
   * Cluster of positive instances
   */
  private ArrayList<ArrayList<Integer>> clusterList;

  /**
   * Cluster directory: index (in the list of clusters) of the cluster
   * containing each positive example.
   */
  private int[] clusterDir;

  /**
   * To store the diverse density information relative to each instance in each
   * positive example. The first index is for the bag and the second for
   * instances inside the bag.
   *
   */
  private double[][] diverseDensityTable;

  /**
   * To calculate instance's diverse density
   */
  private DiverseDensity DDFunction;

  private InstanceLevelDistance cosineDist = new CosineDist();
  private BagLevelDistance aveHausdorffDist = new AveHausdorffDist(cosineDist);
  //private BagLevelDistance aveHausdorffDist = new AveHausdorffDist(new NEuclideanDist());

  /**
   * To store the k nearest neighbors of each positive example.
   */
  private ArrayList<Integer>[] kNN;

  /**
   * Auxiliar member to get information about the class attribute.
   */
  private AttributeStats classStats;

  private int numNNPosFiltering;    // K1
  private int numNNBorderMaj;       // K2
  private int numNNInforMin;        // K3
  private int numNNAttrVoting;      // K4

  public MwmoteDataManager(Instances data, int posClassLabel, int numNNPosFiltering,
          int numNNBorderMaj, int numNNInforMin, int numNNAttrVoting) {
    this.data = data;
    this.posClassLabel = posClassLabel;
    this.numNNAttrVoting = numNNAttrVoting;
    this.numNNPosFiltering = numNNPosFiltering;
    this.numNNBorderMaj = numNNBorderMaj;
    this.numNNInforMin = numNNInforMin;
    // Be aware: order matters
    checkPosClassLabel();
    divideSamplesByClass();
    checkNumNNInforMin();
    findPosPosDistances();
    findPosNegDistances();
    filterPositives();
    findBorderMaj();
    findInforMin();
    computeSelectionProbs();
    findClusters();
    linkSampleToCluster();
    evaluateDD();
    if (hasNominalAttrs()) computeKNN();
  }

  private void checkPosClassLabel() {
    classStats = data.attributeStats(data.classIndex());
    if (posClassLabel == needToBeCalculated) {
      posClassLabel = Utils.minIndex(classStats.nominalCounts);
    }
  }
  
  public MwmoteDataManager(Instances data) {
    this(data, needToBeCalculated, defaultNumNNPosFiltering, defaultNumNNBorderMaj,
            needToBeCalculated, defaultNumNNAttrVoting);
  }

  private void divideSamplesByClass() {
    positives = new ArrayList<Integer>(classStats.nominalCounts[posClassLabel]);
    negatives = new ArrayList<Integer>(data.size() - positives.size());
    for (int i = 0; i < data.numInstances(); i++) {
      if (data.get(i).classValue() == posClassLabel) {
        positives.add(i);
      } else {
        negatives.add(i);
      }
    }
  }

  private void checkNumNNInforMin() {
    if (numNNInforMin == needToBeCalculated) {
      numNNInforMin = positives.size() / 2;
    }
  }

  private void findPosPosDistances() {
    posPosDistances = new double[positives.size()][positives.size()];
    for (int i = 0; i < positives.size(); i++) {
      posPosDistances[i][i] = 1;                      // Do not take into account the distance to itself
      for (int j = 0; j < i; j++) {
        Instance bag1 = data.get(positives.get(i));
        Instance bag2 = data.get(positives.get(j));
        posPosDistances[i][j] = aveHausdorffDist.measure(bag1, bag2);  // Measures.hausdorff_ave(bag1, bag2);
        posPosDistances[j][i] = posPosDistances[i][j];
      }
    }
  }

  private void findPosNegDistances() {
    posNegDistances = new double[positives.size()][negatives.size()];
    for (int i = 0; i < positives.size(); i++) {
      for (int j = 0; j < negatives.size(); j++) {
        Instance bag1 = data.get(positives.get(i));
        Instance bag2 = data.get(negatives.get(j));
        posNegDistances[i][j] = aveHausdorffDist.measure(bag1, bag2);  // Measures.hausdorff_ave(bag1, bag2);
      }
    }
  }

  private void filterPositives() {
    // Step 1 and 2
    posFiltered = new ArrayList<Integer>(positives.size());
    for (int i = 0; i < positives.size(); i++) {
      if (anotherPosInKNN(i, numNNPosFiltering))
        posFiltered.add(i);
    }
  }

  private void findBorderMaj() {
    // Step 3 and 4
    borderMaj = new ArrayList<Integer>(numNNBorderMaj);
    ArrayList<Integer> Nmaj;
    for (int i = 0; i < posFiltered.size(); i++) {
      Nmaj = findNegNN(posFiltered.get(i), numNNBorderMaj);
      borderMaj.removeAll(Nmaj);
      borderMaj.addAll(Nmaj);
    }
  }

  private void findInforMin() {
    // Step 5 and 6
    inforMin = new ArrayList<Integer>(numNNInforMin);
    ArrayList<Integer> Nmin;
    for (int i = 0; i < borderMaj.size(); i++) {
      Nmin = findPosNN(borderMaj.get(i), numNNInforMin);
      inforMin.removeAll(Nmin);
      inforMin.addAll(Nmin);
    }
  }

  private void computeSelectionProbs() {
    // Step 7 and 8
    double[] selWeight = new double[inforMin.size()];
    for (int i = 0; i < borderMaj.size(); i++) {
      for (int j = 0; j < inforMin.size(); j++) {
        double inforWeight = computeWeight(borderMaj.get(i), inforMin.get(j));
        selWeight[j] += inforWeight;
      }
    }

    // Step 9
    double sumWeight = Utils.sum(selWeight);
    Utils.normalize(selWeight, sumWeight);
    selectionProb = selWeight;
  }

  private void findClusters() {
    clusterList = new ArrayList<ArrayList<Integer>>(positives.size());
    ArrayList<Integer> cluster;

    // Initialization
    for (int i = 0; i < positives.size(); i++) {
      cluster = new ArrayList<Integer>(1);
      cluster.add(i);
      clusterList.add(cluster);
    }
    double[][] clusClusDistances = posPosDistances.clone(); // Recuerda dejar claro lo de la diagonal: 0 vs 1

    // Compute the distance threshold for the stop condition based on the average
    // distance between positive examples
  //  double distSum = 0;
  //  for (int i = 0; i < posFiltered.size(); i++) {
  //    distSum += posPosDistances[posFiltered.get(i)][Utils.minIndex(posPosDistances[posFiltered.get(i)])];
  //  }
  //  double distAvg = distSum / posFiltered.size();
  //  double distThreshold = (distAvg + OUTPUT_LEVEL) / (OUTPUT_LEVEL + 1);
    double maxDistance;

    do { // Repeat

      // Find the two closest clusters
      int minX = 0;
      int minY = 0;
      double minDistance = Double.MAX_VALUE;
      for (int i = 0; i < clusterList.size(); i++) {
        for (int j = 0; j < i; j++) {
          if (clusClusDistances[i][j] < minDistance) {
            minDistance = clusClusDistances[i][j];
            minX = j;
            minY = i;
          }
        }
      }

      // Clustering stop condition
      double xNegDist = closerNegToClusterMinDist(clusterList.get(minX));
      double yNegDist = closerNegToClusterMinDist(clusterList.get(minY));
      double maxNegDist = Math.max(xNegDist, yNegDist);
      if (maxNegDist < minDistance) break;

      // Create a new cluster by aggregating the two closest
      int newClusterSize = clusterList.get(minX).size() + clusterList.get(minY).size();
      ArrayList<Integer> newCluster = new ArrayList<Integer>(newClusterSize);
      newCluster.addAll(clusterList.get(minX));
      newCluster.addAll(clusterList.get(minY));

      // Update the cluster list
      // Note that minX and minY are always sorted so that minX < minY
      clusterList.remove(minY);
      clusterList.remove(minX);
      clusterList.add(newCluster);

      // Update the distance matrix
      int uptadedRow = 0;
      double[][] newDistanceMatrix = new double[clusClusDistances.length - 1][];
      for (int i = 0; i < clusClusDistances.length; i++) {
        if (i == minX || i == minY) continue;
        int uptadedCol = 0;
        newDistanceMatrix[uptadedRow] = new double[uptadedRow + 1];
        newDistanceMatrix[uptadedRow][uptadedRow] = 1;
        for (int j = 0; j < i; j++) {
          if (j == minX || j == minY) continue;
          newDistanceMatrix[uptadedRow][uptadedCol++] = clusClusDistances[i][j];
        }
        uptadedRow++;
      }
      newDistanceMatrix[clusterList.size() - 1] = new double[clusterList.size()];
      maxDistance = 0;
      for (int i = 0; i < clusterList.size() - 1; i++) {
        cluster = clusterList.get(i);
        double dist = clusterDistance(newCluster, cluster);
        newDistanceMatrix[clusterList.size() - 1][i] = dist;
        if (dist > maxDistance)
          maxDistance = dist;
      }
      clusClusDistances = newDistanceMatrix;
    } while (clusterList.size() > 1);
  //  } while (maxDistance <= distThreshold);

    double avgClusterSize = 0;    // Debug
    for (int i = 0; i < clusterList.size(); i++) {
      cluster = clusterList.get(i);
      avgClusterSize += cluster.size(); // Debug
    }
    avgClusterSize /= clusterList.size(); // Debug
    System.out.println("Number of clusters: " + clusterList.size());  // Debug
    System.out.println("Average cluster size: " + avgClusterSize);  // Debug
  }

  private void linkSampleToCluster() {
    clusterDir = new int[positives.size()];
    for (int i = 0; i < clusterList.size(); i++) {
      ArrayList<Integer> cluster = clusterList.get(i);
      for (int j = 0; j < cluster.size(); j++) {
        clusterDir[cluster.get(j)] = i;
      }
    }
  }

  private void evaluateDD() {
    DDFunction = new DiverseDensity(data, posClassLabel);
    diverseDensityTable = new double[positives.size()][];
    for (int i = 0; i < positives.size(); i++) {
      Instance bag = data.get(positives.get(i));
      int bagSize = bag.relationalValue(1).size();
      diverseDensityTable[i] = new double[bagSize];
      for (int j = 0; j < bagSize; j++) {
        Instance x = bag.relationalValue(1).get(j);
//        diverseDensityTable[i][j] = DDFunction.noisyOr(x);
        diverseDensityTable[i][j] = DDFunction.mostLikelyCause(x);
      }
    }
  }

  /**
   * Find the K nearest positive neighbors to a given positive example.
   *
   * @param posIndex index of the given positive example in the list of postive
   * examples indexes.
   * @param K number of nearest neightbors to return.
   * @return a list with indexes for the K nearest neightbors.
   */
  private ArrayList<Integer> getKNN(int posIndex, int K) {
    ArrayList<Integer> NN = new ArrayList<Integer>(K);
    double[] posDist = posPosDistances[posIndex];
    int[] sortedPosDist = Utils.sort(posDist);
    for (int i = 0; i < K; i++) {
      NN.add(sortedPosDist[i]);
    }
    return NN;
  }

  private void computeKNN() {
    kNN = new ArrayList[positives.size()];
    for (int i = 0; i < positives.size(); i++) {
      kNN[i] = getKNN(i, numNNAttrVoting);
    }
  }

  /**
   * Determines whether a positive example has another positive example among his
   * K nearest neighbors.
   * @param posIndex selIndex of the example in the list of positive examples
   * @param K number of nearest neighbors to be considered
   * @return true if the example has at least one positive example among its K
   * nearest neighbors, otherwise returns false.
   */
  private boolean anotherPosInKNN(int posIndex, int K) {
    double[] posDist = posPosDistances[posIndex];
    double minPosDist = posDist[Utils.minIndex(posDist)];
    double[] negDist = posNegDistances[posIndex];
    int[] sortedNegDist = Utils.sort(negDist);      // Efficiency issue
    double KMinNegDist = negDist[sortedNegDist[K - 1]];
    return minPosDist < KMinNegDist;
  }

  /**
   * Finds the K negative nearest neighbors to a given positive instance.
   * @param posIndex selIndex in the list of positive examples of the example that we
   * want to get the nearest neighbors
   * @param positives list of selIndex in training dataset of positive examples
   * @param negatives list of selIndex in training dataset of negative examples
   * @param K number of nearest neighbors to be returned
   * @param posNegDistances matrix of posPosDistances between positive and negative examples
   * @return a list with the K nearest negative neighbor indexes in the list of
   * negative examples.
   */
  private ArrayList<Integer> findNegNN(int posIndex, int K) {
    // Distances from the target positive example to each negative example
    double[] negDist = posNegDistances[posIndex];
    // Indexes of the posPosDistances sorted in ascending order
    int[] sortedNegDist = Utils.sort(negDist);              // Efficiency issue
    // Creates an empty set for holding the KNN
    ArrayList<Integer> negNN = new ArrayList<Integer>(K);
    // Adds selIndex in training dataset of negative examples which
    // distance to the target positive example is among the K lesser
    for (int i = 0; i < K; i++) {
      negNN.add(sortedNegDist[i]);
    }
    return negNN;
  }

  /**
   * Finds the K positive nearest neighbors to a given negative instance.
   *
   * @param negIndex selIndex in the list of negative examples of the example that we
   * want to get the nearest neighbors
   * @param K number of nearest neighbors to be returned
   * @return a list with the K nearest positive neighbor indexes in the list of
   * positive examples.
   */
  private ArrayList<Integer> findPosNN(int negIndex, int K) {
    // Distances from the target negative example to each positive example
    double[] posDist = new double[positives.size()];
    for (int i = 0; i < positives.size(); i++) {
      posDist[i] = posNegDistances[i][negIndex];
    }
    // Indexes of the posPosDistances sorted in ascending order
    int[] sortedPosDist = Utils.sort(posDist);              // Efficiency issue
    // Creates an empty set for holding the KNN
    ArrayList<Integer> posNN = new ArrayList<Integer>(K);
    // Adds selIndex in training dataset of positive examples which
    // distance to the target negative example is among the K lesser
    for (int i = 0; i < K; i++) {
      posNN.add(sortedPosDist[i]);
    }
    return posNN;
  }

  /**
   * Calculates the distance between a given positive example cluster and its closer
   * negative example as the minimal distance between the negative example and each
   * positive example in the cluster.
   *
   * @param cluster
   * @return
   */
  private double closerNegToClusterMinDist(ArrayList<Integer> cluster) {
    double minDist = Double.MAX_VALUE;
    for (int i = 0; i < cluster.size(); i++) {
      int posXindex = cluster.get(i);
      int minPosXindex = Utils.minIndex(posNegDistances[posXindex]);
      double minPosX = posNegDistances[posXindex][minPosXindex];
      if (minPosX < minDist)
        minDist = minPosX;
    }
    return minDist;
  }

  /**
   * Calculates the distance between a given positive example cluster and its closer
   * negative example as the average distance between the negative example and each
   * positive example in the cluster.
   *
   * @param cluster
   * @return
   */
  private double closerNegToClusterAvgDist(ArrayList<Integer> cluster) {
    double minDist = Double.MAX_VALUE;
    for (int i = 0; i < borderMaj.size(); i++) {
      double dist = 0;
      for (int j = 0; j < cluster.size(); j++) {
        dist += posNegDistances[cluster.get(j)][borderMaj.get(i)];
      }
      dist /= cluster.size();
      if (dist < minDist)
        minDist = dist;
    }
    return minDist;
  }

/**
 * Calculates the minimun posPosDistances between a given positive example and a
 * positive example cluster. It is used in the Hausdorff distance calculation.
 *
 * @param c1Indes index of the given positive example in the positive example list
 * @param c2 the positive example cluster
 *
 * @return the distance.
 */
  private double min(int c1Index, ArrayList<Integer> c2) {
    double m = Double.MAX_VALUE;
    for (int i = 0; i < c2.size(); i++) {
      int c2Index = c2.get(i);
      double s = posPosDistances[c1Index][c2Index];
      //double s = NEuclidean(x, a);
      if (s < m)
        m = s;
    }
    return m;
  }

/**
 * Calculates the maximum of the minimun posPosDistances between two given positive
 * example clusters used in the Hausdorff distance calculation.
 *
 * @param c1 the first cluster
 * @param c2 the second cluster
 *
 * @return the distance.
 */
  private double max_min(ArrayList<Integer> c1, ArrayList<Integer> c2) {
    double m = 0;
    for (int i = 0; i < c1.size(); i++) {
      int c1Index = c1.get(i);
      double s = min(c1Index, c2);
      if (s == 1) return 1;
      if (s > m)
        m = s;
    }
    return m;
  }

  /**
   * Calculates the average Hausdorff posPosDistances between two given positive example
   * clusters.
   *
   * @param c1 the first cluster
   * @param c2 the second cluster
   *
   * @return the distance.
   */
  private double hausdorff_ave(ArrayList<Integer> c1, ArrayList<Integer> c2) {
    double m = 0;
    for (int i = 0; i < c1.size(); i++) {
      int c1Index = c1.get(i);
      m += min(c1Index, c2);
    }
    for (int i = 0; i < c2.size(); i++) {
      int c2Index = c2.get(i);
      m += min(c2Index, c1);
    }
    int N = c1.size() + c2.size();
    m /= N;
    return m;
  }


  /**
   * Calculates the maximal Hausdorff posPosDistances between two given positive example
   * clusters.
   *
   * @param c1 the first cluster
   * @param c2 the second cluster
   *
   * @return the distance.
   */
  private double hausdorff(ArrayList<Integer> c1, ArrayList<Integer> c2) {
    double n1 = max_min(c1, c2);
    if (n1 == 1) return 1;
    double n2 = max_min(c2, c1);
    if (n1 > n2)
      return n1;
    else
      return n2;
  }

  /**
   * Calculates the posPosDistances between two given positive example clusters using
   * any of the Hausdorff distance variants.
   *
   * @param c1 the first cluster
   * @param c2 the second cluster
   *
   * @return the distance.
   */
  private double clusterDistance(ArrayList<Integer> c1, ArrayList<Integer> c2) {
    //return hausdorff(c1, c2);
    return hausdorff_ave(c1, c2);
  }

  /**
   * Computes the importance of a given pair of (positive and negative) examples
   * to generate synthetic examples.
   *
   * @param negIndex selIndex in the list of negative examples of the negative example
   * in the pair that we want to compute the weight.
   * @param posIndex selIndex in the list of positive examples of the positive example
   * in the pair that we want to compute the weight.
   * @return the weight of the given pair of examples.
   */
  private double computeWeight(int negIndex, int posIndex) {
    double cSum = 0;
    for (int i = 0; i < inforMin.size(); i++) {
      cSum += posNegDistances[inforMin.get(i)][negIndex];
    }
    double closeness = posNegDistances[posIndex][negIndex];
    double density = closeness / cSum;
    return closeness * density;
  }

  private boolean hasNominalAttrs() {
    /**
     * Header info for the bag
     */
    Instances bagHeader = data.attribute(1).relation();
    Enumeration attrEnum = bagHeader.enumerateAttributes();
    while(attrEnum.hasMoreElements()) {
      Attribute attr = (Attribute) attrEnum.nextElement();
      if (attr.isNominal()) {
        return true;
      }
    }
    return false;
  }

  /**
   *               PUBLIC CLAUSES
   */

  /**
   * Gets the number of neighbors used for predicting noisy minority class samples.
   * It is a parameters of the algorithm.
   */

  public int numNNPosFiltering() {
    return numNNPosFiltering;
  }

  /**
   * Sets the number of neighbors used for predicting noisy minority class samples.
   *
   * @param K the new value.
   */
  public void setNumNNPosFiltering(int K) {
    numNNPosFiltering = K;
  }

  /**
   *  Gets the number of majority neighbors used for constructing informative
   * minority set. It is a parameters of the algorithm.
   */
  public int numNNBorderMaj() {
    return numNNBorderMaj;
  }

  /**
   * Sets the number of majority neighbors used for constructing informative
   * minority set.
   *
   * @param K the new value.
   */
  public void setNumNNBorderMaj(int K) {
    numNNBorderMaj = K;
  }

  /**
   * Gets the number of minority neighbors used for constructing informative
   * minority set. It is a parameters of the algorithm.
   */
  public int numNNInforMin() {
    return numNNInforMin;
  }

  /**
   * Sets the number of minority neighbors used for constructing informative
   * minority set.
   *
   * @param K the new value.
   */
  public void numNNInforMin(int K) {
    numNNInforMin = K;
  }

  /**
   * Gets the number of minority neighbors used for determining nominal attribute
   * values. It is a parameters of the algorithm.
   */
  public int numNNAttrVoting() {
    return numNNAttrVoting;
  }

  /**
   * Sets the number of minority neighbors used for determining nominal attribute
   * values.
   *
   * @param K the new value.
   */
  public void setNumNNAttrVoting(int K) {
    numNNAttrVoting = K;
  }

  /**
   * Gets the index of the positive class label
   */
  public int posClassLabel() {
    return posClassLabel;
  }

  /**
   * Gets the list of positive example indexes in the training set.
   */
  public ArrayList<Integer> positives() {
    return positives;
  }

  /**
   * Gets the list of negative example indexes in the training set.
   */
  public ArrayList<Integer> negatives() {
    return negatives;
  }

  /**
   * Size of the positive (or minority) class.
   *
   * @return the number of positive examples in the dataset.
   */
  public int posCount() {
    return positives.size();
  }

  /**
   * Size of the negative (or majority) class.
   *
   * @return the number of negative examples in the dataset.
   */
  public int negCount() {
    return negatives.size();
  }

  /**
   * Gets the i-th positive example in the dataset.
   *
   * @param i position relative to the list of positive examples.
   *
   * @return data.get(positives.get(i));
   */
  public Instance posBag(int i) {
    return data.get(positives.get(i));
  }

  /**
   * Gets the i-th negative example in the dataset.
   *
   * @param i position relative to the list of negative examples.
   *
   * @return data.get(negatives.get(i));
   */
  public Instance negBag(int i) {
    return data.get(negatives.get(i));
  }

  /**
   * Gets the list of informative positive examples indexes in the list of
   * positive examples.
   */
  public ArrayList<Integer> inforMin() {
    return inforMin;
  }

  /**
   * Gets the size of the list of informative positive examples indexes in the list of
   * positive examples.
   *
   * @return inforMin.size();
   */
  public int inforMinSize() {
    return inforMin.size();
  }

  /**
   * Gets the i-th index in the list of positive examples.
   *
   * @param i position relative to the list of positive example indexes.
   *
   * @return inforMin.get(i);
   */
  public int inforMin(int i) {
    return inforMin.get(i);
  }

  /**
   * Gets the cluster containing the example with i i in the list of positive
   * examples.
   * @param i position of the example in the list of positive examples.
   *
   * @return clusterList.get(clusterDir[i]);
   */
  public ArrayList<Integer> clusterWithPosBag(int i) {
    return clusterList.get(clusterDir[i]);
  }

  /**
   * Gets the diverse density value of an instance inside an example (bag).
   *
   * @param posBagIndex index in the list of positive examples of the bag
   * containing the instance.
   * @param instIndex the index of the instance inside the bag.
   *
   * @return A BigDecimal representation of the diverse density value.
   */
  public double diverseDensity(int posBagIndex, int instIndex) {
    return diverseDensityTable[posBagIndex][instIndex];
  }

  /**
   * Gets the diverse density values of all the instances in an example (bag).
   * @param posBagIndex index in the list of positive examples of the bag 
   * containing the instance.
   * @return an array of BogDecimal representing the diverse densiti values.
   */
  public double[] diverseDensity(int posBagIndex) {
    return diverseDensityTable[posBagIndex];
  }

  /**
   * Gets a list with the nearest neighbors of a positive example.
   *
   * @param i index of the example in the list of positive examples.
   *
   * @return the kNN as an ArrayList<Integer>.
   */
  public ArrayList<Integer> NNToPos(int i) {
    return kNN[i];
  }

  /**
   * Gets the selection probability of all informative positive example.
   * @return an array of doubles with the probabilities.
   */
  public double[] selectionProb() {
    return selectionProb;
  }

}
