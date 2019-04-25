/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import java.util.Arrays;

/**
 *
 * @author Danel
 */
public class OWA {
  
  public static final int vw_CLASSIC_MAX = 1;
  public static final int vw_LINEAL_MAX = 3;
  public static final int vw_LINEAL_MIN = 4;
  public static final int vw_POW2_MAX = 5;
  public static final int vw_HYPERBOL_MAX = 7;

  /**
   * The weight vector
   */
  private double[] weights;

  /**
   * index of each operand in the ascending sorting.
   */
   private int[] operandSortedIndex;

  
  public OWA(int vectorWeighing, int vectorSize) {
    weights = new double[vectorSize];
    switch (vectorWeighing) {
      case vw_CLASSIC_MAX:
        makeClassicMax();
      case vw_LINEAL_MAX:
        makeLinealMax();
        break;
      case vw_LINEAL_MIN:
        makeLinealMin();
        break;
      case vw_POW2_MAX:
        makePow2Max();
        break;
      case vw_HYPERBOL_MAX:
        makeHyperbolMax();
        break;
    }
  }

  private void makeClassicMax() {
    Arrays.fill(weights, 0);
    weights[0] = 1;
  }
  
  private void makeLinealMax() {
    double p = weights.length;
    double Z = p * (p + 1);
    for (int i = 1; i <= p; i++) {
      weights[i - 1] = 2 * (p - i + 1) / Z;
    }
  }

  private void makeLinealMin() {
    double p = weights.length;
    double Z = p * (p + 1);
    for (int i = 1; i <= p; i++) {
      weights[i - 1] = 2 * i / Z;
    }
  }

  private void makePow2Max() {
    double p = weights.length;
    double Z = Math.pow(2, p) - 1;
    for (int i = 1; i <= p; i++) {
      weights[i - 1] = Math.pow(2, p - i) / Z;
    }
  }

  private void makeHyperbolMax() {
    double p = weights.length;
    double Z = 0;
    for (int j = 1; j <= p; j++) {
      Z += 1 / j;
    }
    for (int i = 1; i <= p; i++) {
      weights[i - 1] = 1 / ( i * Z);
    }
  }

  /**
   * Gets the value of the operand in the position "index" as sorted in descending order.
   * @param index integer between 0 and the number of operands minus one.
   * @return the value of the operand.
   */
  private double decreasingOperandValue(int index, double[] operands) {
    return operands[operandSortedIndex[operandSortedIndex.length - index - 1]];
  }

  /**
   * Sort the operands in ascending order.
   */
  private void sortOperands(double[] operands) {
    operandSortedIndex = weka.core.Utils.sort(operands);
  }

  public double operate(double[] operands) {
    if (operands.length != weights.length)
      throw new IllegalArgumentException("operands.length != weights.length");
    sortOperands(operands);
    double sum = 0;
    for (int i = 0; i < operands.length; i++) {
      sum += weights[i] * decreasingOperandValue(i, operands);
    }
    return sum;
  }

}
