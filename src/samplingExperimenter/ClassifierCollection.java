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
public class ClassifierCollection {

  private ArrayList<ClassifierEntry> classifierEntries;

  public ClassifierCollection() {
    classifierEntries = new ArrayList<ClassifierEntry>();
    BookCollection();
  }

  private void singleInstanceWrappers() {
    classifierEntries.add(new ClassifierEntry("StdWrapper(C4.5 C=0.025)", "samplingExperimenter.MIStdWrapper -A 3 -W weka.classifiers.trees.J48 -- -C 0.025 -M 2"));
    classifierEntries.add(new ClassifierEntry("ColWrapper(C4.5 C=0.025)", "weka.classifiers.mi.MIWrapper -P 2 -A 3 -W weka.classifiers.trees.J48 -- -C 0.025 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MIStdWrapper (LogReg)", "samplingExperimenter.MIStdWrapper -A 3 -W weka.classifiers.functions.Logistic -- -C -R 1.0E-8 -M -1"));
//    classifierEntries.add(new ClassifierEntry("MIColWrapper (LogReg)", "weka.classifiers.mi.MIWrapper -P 2 -A 3 -W weka.classifiers.functions.Logistic -- -C -R 1.0E-8 -M -1"));
//    classifierEntries.add(new ClassifierEntry("MIStdWrapper (SVM RBF (C=10, G=0.5))", "samplingExperimenter.MIStdWrapper -A 3 -W weka.classifiers.functions.SMO -- -C 10.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("MIColWrapper (SVM RBF (C=10, G=0.5))", "weka.classifiers.mi.MIWrapper -P 2 -A 3 -W weka.classifiers.functions.SMO -- -C 10.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
  }
  
  private void BookCollection() {
    classifierEntries.add(new ClassifierEntry("CCE(1NN)", "classifiers.CCE -I 5 -W weka.classifiers.lazy.IBk -- -K 1 -W 0"));
    classifierEntries.add(new ClassifierEntry("CCE(C4.5)", "classifiers.CCE -I 5 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("CCE(LogReg)", "classifiers.CCE -I 5 -W weka.classifiers.functions.Logistic -- -R 1.0E-8 -M -1"));
    classifierEntries.add(new ClassifierEntry("CCE(SVM C=10 G=0.5)", "classifiers.CCE -I 5 -W weka.classifiers.functions.SMO -- -C 10.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
    classifierEntries.add(new ClassifierEntry("CCE(AdaBoost+C4.5)", "classifiers.CCE -I 5 -W weka.classifiers.meta.AdaBoostM1 -- -P 100 -S 1 -I 10 -W weka.classifiers.trees.J48 -- -C 0.025 -M 2"));

    classifierEntries.add(new ClassifierEntry("ConcatMap(1NN)", "classifiers.ConcatMap -I 5 -W weka.classifiers.lazy.IBk -- -K 1 -W 0"));
    classifierEntries.add(new ClassifierEntry("ConcatMap(C4.5)", "classifiers.ConcatMap -I 5 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("ConcatMap(LogReg)", "classifiers.ConcatMap -I 5 -W weka.classifiers.functions.Logistic -- -R 1.0E-8 -M -1"));
    classifierEntries.add(new ClassifierEntry("ConcatMap(SVM C=10 G=0.5)", "classifiers.ConcatMap -I 5 -W weka.classifiers.functions.SMO -- -C 10.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
    classifierEntries.add(new ClassifierEntry("ConcatMap(AdaBoost+C4.5)", "classifiers.ConcatMap -I 5 -W weka.classifiers.meta.AdaBoostM1 -- -P 100 -S 1 -I 10 -W weka.classifiers.trees.J48 -- -C 0.025 -M 2"));
  }

  private void BagBasedCollection() {
//    classifierEntries.add(new ClassifierEntry("CitationKNN(1,1)", "weka.classifiers.mi.CitationKNN -R 1 -C 1 -H 1"));
//    classifierEntries.add(new ClassifierEntry("CitationKNN(1,2) ", "weka.classifiers.mi.CitationKNN -R 1 -C 2 -H 1"));
//    classifierEntries.add(new ClassifierEntry("CitationKNN(2,1)", "weka.classifiers.mi.CitationKNN -R 2 -C 1 -H 1"));
//    classifierEntries.add(new ClassifierEntry("CitationKNN(2,2)", "weka.classifiers.mi.CitationKNN -R 2 -C 2 -H 1"));
//    classifierEntries.add(new ClassifierEntry("CitationKNN(2,3)", "weka.classifiers.mi.CitationKNN -R 2 -C 3 -H 1"));
//    classifierEntries.add(new ClassifierEntry("CitationKNN(3,2)", "weka.classifiers.mi.CitationKNN -R 3 -C 2 -H 1"));
//    classifierEntries.add(new ClassifierEntry("CitationKNN(3,3)", "weka.classifiers.mi.CitationKNN -R 3 -C 3 -H 1"));
//    classifierEntries.add(new ClassifierEntry("MISMO(K1 C=0.1)", "weka.classifiers.mi.MISMO -C 0.1 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 1.0\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(K1 C=1.0)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 1.0\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(K1 C=10)", "weka.classifiers.mi.MISMO -C 10.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 1.0\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(K2 C=0.1)", "weka.classifiers.mi.MISMO -C 0.1 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 2.0\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(K2 C=1.0)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 2.0\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(K2 C=10)", "weka.classifiers.mi.MISMO -C 10.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 2.0\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(RBF C=0.1,G=0.01)", "weka.classifiers.mi.MISMO -C 0.1 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIRBFKernel -C 250007 -G 0.01\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(RBF C=0.1,G=0.1)", "weka.classifiers.mi.MISMO -C 0.1 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIRBFKernel -C 250007 -G 0.1\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(RBF C=0.1,G=0.5)", "weka.classifiers.mi.MISMO -C 0.1 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIRBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(RBF C=1,G=0.01)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIRBFKernel -C 250007 -G 0.01\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(RBF C=1,G=0.1)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIRBFKernel -C 250007 -G 0.1\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(RBF C=1,G=0.5)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIRBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(RBF C=10,G=0.01)", "weka.classifiers.mi.MISMO -C 10.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIRBFKernel -C 250007 -G 0.01\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(RBF C=10,G=0.1)", "weka.classifiers.mi.MISMO -C 10.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIRBFKernel -C 250007 -G 0.1\""));
//    classifierEntries.add(new ClassifierEntry("MISMO(RBF C=10,G=0.5)", "weka.classifiers.mi.MISMO -C 10.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIRBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("SimpleMI (A, KNN K=1)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.lazy.IBk -- -K 1 -W 0"));
//    classifierEntries.add(new ClassifierEntry("SimpleMI (G, KNN K=1)", "weka.classifiers.mi.SimpleMI -M 2 -W weka.classifiers.lazy.IBk -- -K 1 -W 0"));
//    classifierEntries.add(new ClassifierEntry("SimpleMI (A, KNN K=2)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.lazy.IBk -- -K 2 -W 0"));
//    classifierEntries.add(new ClassifierEntry("SimpleMI (G, KNN K=2)", "weka.classifiers.mi.SimpleMI -M 2 -W weka.classifiers.lazy.IBk -- -K 2 -W 0"));
//    classifierEntries.add(new ClassifierEntry("SimpleMI (A, KNN K=3)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.lazy.IBk -- -K 3 -W 0"));
//    classifierEntries.add(new ClassifierEntry("SimpleMI (G, KNN K=3)", "weka.classifiers.mi.SimpleMI -M 2 -W weka.classifiers.lazy.IBk -- -K 3 -W 0"));
//    classifierEntries.add(new ClassifierEntry("SimpleMI (A, C4.5)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("SimpleMI (G, C4.5)", "weka.classifiers.mi.SimpleMI -M 2 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("SimpleMI(A, LogReg)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.functions.Logistic -- -R 1.0E-8 -M -1"));
//    classifierEntries.add(new ClassifierEntry("SimpleMI(A, Boosting+DStump)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.meta.AdaBoostM1 -- -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump"));
//    classifierEntries.add(new ClassifierEntry("SimpleMI(A, Boosting+C4.5)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.meta.AdaBoostM1 -- -P 100 -S 1 -I 10 -W weka.classifiers.trees.J48 -- -C 0.025 -M 2"));
//    classifierEntries.add(new ClassifierEntry("Simple(A,SVM C=0.1 G=0.01)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.functions.SMO -- -C 0.1 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\""));
//    classifierEntries.add(new ClassifierEntry("Simple(A,SVM C=0.1 G=0.1)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.functions.SMO -- -C 0.1 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.1\""));
//    classifierEntries.add(new ClassifierEntry("Simple(A,SVM C=0.1 G=0.5)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.functions.SMO -- -C 0.1 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("Simple(A,SVM C=1 G=0.01)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.functions.SMO -- -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\""));
//    classifierEntries.add(new ClassifierEntry("Simple(A,SVM C=1 G=0.1)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.functions.SMO -- -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.1\""));
//    classifierEntries.add(new ClassifierEntry("Simple(A,SVM C=1 G=0.5)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.functions.SMO -- -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("Simple(A,SVM C=10 G=0.01)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.functions.SMO -- -C 10.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\""));
//    classifierEntries.add(new ClassifierEntry("Simple(A,SVM C=10 G=0.1)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.functions.SMO -- -C 10.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.1\""));
//    classifierEntries.add(new ClassifierEntry("Simple(A,SVM C=10 G=0.5)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.functions.SMO -- -C 10.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 default)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 894.4271909999159\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=500)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 500\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=250)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 250\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=125)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 125\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=60)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 60\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=30)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 30\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=10)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 10\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=5)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 5\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=1)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 1\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=0.5)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 0.5\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=0.1)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 0.1\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=0.01)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 0.01\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES (C4.5 s=0.001)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 0.001\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MILES(1NN)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 250\" -W weka.classifiers.lazy.IBk -- -K 1 -W 0"));
//    classifierEntries.add(new ClassifierEntry("MILES(LogReg)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 250\" -W weka.classifiers.functions.Logistic -- -R 1.0E-8 -M -1"));
//    classifierEntries.add(new ClassifierEntry("MILES(SVM C=1 G=0.5)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 250\" -W weka.classifiers.functions.SMO -- -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("MILES(SVM C=10 G=0.5)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 250\" -W weka.classifiers.functions.SMO -- -C 10.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("MILES(AdaBoost+C4.5)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 250\" -W weka.classifiers.meta.AdaBoostM1 -- -P 100 -S 1 -I 10 -W weka.classifiers.trees.J48 -- -C 0.025 -M 2"));
//    classifierEntries.add(new ClassifierEntry("BARTMIP(1NN)", "classifiers.Bartmip -W weka.classifiers.lazy.IBk -- -K 1 -W 0"));
//    classifierEntries.add(new ClassifierEntry("BARTMIP(C4.5)", "classifiers.Bartmip -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
//    classifierEntries.add(new ClassifierEntry("BARTMIP(LogReg)", "classifiers.Bartmip -W weka.classifiers.functions.Logistic -- -R 1.0E-8 -M -1"));
//    classifierEntries.add(new ClassifierEntry("BARTMIP(SVM C=1 G=0.5)", "classifiers.Bartmip -W weka.classifiers.functions.SMO -- -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("BARTMIP(SVM C=10 G=0.5)", "classifiers.Bartmip -W weka.classifiers.functions.SMO -- -C 10.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("BARTMIP(AdaBoost+C4.5)", "classifiers.Bartmip -W weka.classifiers.meta.AdaBoostM1 -- -P 100 -S 1 -I 10 -W weka.classifiers.trees.J48 -- -C 0.025 -M 2"));
//    classifierEntries.add(new ClassifierEntry("CCE(C4.5)", "classifiers.CCE -I 5 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("ConcatMap(C4.5)", "classifiers.ConcatMap -I 5 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
  }

  private void InstanceBasedCollection() {
    /**
     * Logistic regression
     */
//    classifierEntries.add(new ClassifierEntry("MILR (G)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 2"));
////    classifierEntries.add(new ClassifierEntry("MILR (S)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 0"));
    /**
     * Diverse density based
     */
//    classifierEntries.add(new ClassifierEntry("EMDD", "weka.classifiers.mi.MIEMDD -S 1 -N 1"));
    /**
     * Trees
     */
//    classifierEntries.add(new ClassifierEntry("MITI", "weka.classifiers.mi.MITI -K 5 -U -B -Ba 0.5 -M 2 -A 1 -An 1 -S 1"));
    /**
     * Boosting
     */
//    classifierEntries.add(new ClassifierEntry("MIBoost (C4.5 C=0.01)", "weka.classifiers.mi.MIBoost -R 10 -B 0 -W weka.classifiers.trees.J48 -- -C 0.01 -M 2"));
//    classifierEntries.add(new ClassifierEntry("MIBoost (C4.5 C=0.005)", "weka.classifiers.mi.MIBoost -R 10 -B 0 -W weka.classifiers.trees.J48 -- -C 0.005 -M 2"));
//    classifierEntries.add(new ClassifierEntry("AdaBoost (MITI)", "weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.mi.MITI -- -K 5 -U -B -Ba 0.5 -M 2 -A 1 -An 1 -S 1"));
    /**
     * Support vector machines
     */
//    classifierEntries.add(new ClassifierEntry("miSVM (RBF C=0.1 G=0.5)", "weka.classifiers.mi.MISVM -C 0.1 -N 0 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("miSVM (RBF C=1 G=0.5)", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("miSVM (RBF C=10 G=0.5)", "weka.classifiers.mi.MISVM -C 10.0 -N 0 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.5\""));
//    classifierEntries.add(new ClassifierEntry("miSVM (K1 C=0.1)", "weka.classifiers.mi.MISVM -C 0.1 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
//    classifierEntries.add(new ClassifierEntry("miSVM (K2)", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 2.0\""));
  }

  private void samplingEnsambleMethods() {
    classifierEntries.add(new ClassifierEntry("AdaBoostM1+MITI", "weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.mi.MITI -- -K 5 -Ba 0.5 -M 2 -A -1 -An 1 -S 1"));
    classifierEntries.add(new ClassifierEntry("MISORUBoostM2+MITI", "ensembles.MISORUBoostM2 -P 100 -S 1 -I 10 -W weka.classifiers.mi.MITI -- -K 5 -Ba 0.5 -M 2 -A -1 -An 1 -S 1"));
  }

  private void collectionToTestSamplingMethods() {
    classifierEntries.add(new ClassifierEntry("MITI", "weka.classifiers.mi.MITI -K 5 -Ba 0.5 -M 2 -A -1 -An 1 -S 1"));
    classifierEntries.add(new ClassifierEntry("CitationKNN", "weka.classifiers.mi.CitationKNN -R 3 -C 3 -H 1"));
//    classifierEntries.add(new ClassifierEntry("SVM", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));                 // safer version
    //classifierEntries.add(new ClassifierEntry("SVM", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -no-checks -C 250007 -E 1.0\""));    // faster version
  }


  /**
   * Almost a full collection of weka MIL classifiers
   */
  private void compileCollection1() {
    /**
     * KNN
     */
    //classifierEntries.add(new ClassifierEntry("CitationKNN (R1,C1)", "weka.classifiers.mi.CitationKNN -R 1 -C 1 -H 1"));
    //classifierEntries.add(new ClassifierEntry("CitationKNN (R3,C3)", "weka.classifiers.mi.CitationKNN -R 3 -C 3 -H 1"));

    /**
     * Trees
     */
    //classifierEntries.add(new ClassifierEntry("MITI", "weka.classifiers.mi.MITI -K 5 -Ba 0.5 -M 2 -A -1 -An 1 -S 1"));

    /**
     * Rules
     */
    //classifierEntries.add(new ClassifierEntry("MIRI", "weka.classifiers.mi.MIRI -K 5 -Ba 0.5 -M 2 -A -1 -An 1 -S 1"));
    
    /**
     * Diverse density
     */
    classifierEntries.add(new ClassifierEntry("MDD", "weka.classifiers.mi.MDD -N 0"));
    classifierEntries.add(new ClassifierEntry("MIDD", "weka.classifiers.mi.MIDD -N 0"));
    classifierEntries.add(new ClassifierEntry("QuickDDIterative", "weka.classifiers.mi.QuickDDIterative -N 0 -S 1.0 -M 1.0 -I 2"));
    
    /**
     * Simple wrappers
     */
    classifierEntries.add(new ClassifierEntry("SimpleMI (A, C4.5)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("SimpleMI (G, C4.5)", "weka.classifiers.mi.SimpleMI -M 2 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (A, C4.5)", "weka.classifiers.mi.MIWrapper -P 1 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (G, C4.5)", "weka.classifiers.mi.MIWrapper -P 2 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (M, C4.5)", "weka.classifiers.mi.MIWrapper -P 3 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    
    /**
     * Wrapper with attribute space mapping
     */
    classifierEntries.add(new ClassifierEntry("MILES (C4.5)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 894.4271909999159\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));

    /**
     * Boosting
     */
    classifierEntries.add(new ClassifierEntry("MIBoost (C4.5)", "weka.classifiers.mi.MIBoost -R 10 -B 0 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("AdaBoost (MITI)", "weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.mi.MITI -- -K 5 -Ba 0.5 -M 2 -A -1 -An 1 -S 1"));
    classifierEntries.add(new ClassifierEntry("AdaBoost (MIRI)", "weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.mi.MIRI -- -K 5 -Ba 0.5 -M 2 -A -1 -An 1 -S 1"));
    
    /**
     * Logistic regression
     */
    classifierEntries.add(new ClassifierEntry("MILR (A)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 1"));
    classifierEntries.add(new ClassifierEntry("MILR (G)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 2"));
    classifierEntries.add(new ClassifierEntry("MILR (S)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 0"));

    /**
     * Support vector machines
     */
    classifierEntries.add(new ClassifierEntry("MISVM (K1)", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
    classifierEntries.add(new ClassifierEntry("MISVM (K2)", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 2.0\""));
    classifierEntries.add(new ClassifierEntry("MISMO (K1)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 1.0\""));
    classifierEntries.add(new ClassifierEntry("MISMO (K2)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 2.0\""));
  }

  /**
   * Dedicated compilation for Suramin dataset.
   * Almost a full collection of weka MIL classifiers but
   * MITI based classifiers have special settings for Suramin dataset.
   */
  private void compileCollection2() {
    /**
     * KNN
     */
    classifierEntries.add(new ClassifierEntry("CitationKNN (R1,C1)", "weka.classifiers.mi.CitationKNN -R 1 -C 1 -H 1"));
    classifierEntries.add(new ClassifierEntry("CitationKNN (R3,C3)", "weka.classifiers.mi.CitationKNN -R 3 -C 3 -H 1"));

    /**
     * Trees
     */
    classifierEntries.add(new ClassifierEntry("MITI", "weka.classifiers.mi.MITI -K 5 -U -B -Ba 0.5 -M 2 -A 1 -An 1 -S 1"));

    /**
     * Rules
     */
    classifierEntries.add(new ClassifierEntry("MIRI", "weka.classifiers.mi.MIRI -K 5 -U -B -Ba 0.5 -M 2 -A 1 -An 1 -S 1"));

    /**
     * Diverse density
     */
    classifierEntries.add(new ClassifierEntry("MINND", "weka.classifiers.mi.MINND -K 1 -S 1 -E 1"));
    classifierEntries.add(new ClassifierEntry("MDD", "weka.classifiers.mi.MDD -N 0"));
    classifierEntries.add(new ClassifierEntry("MIDD", "weka.classifiers.mi.MIDD -N 0"));
    classifierEntries.add(new ClassifierEntry("QuickDDIterative", "weka.classifiers.mi.QuickDDIterative -N 0 -S 1.0 -M 1.0 -I 2"));

    /**
     * Simple wrappers
     */
    classifierEntries.add(new ClassifierEntry("SimpleMI (A, C4.5)", "weka.classifiers.mi.SimpleMI -M 1 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("SimpleMI (G, C4.5)", "weka.classifiers.mi.SimpleMI -M 2 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (A, C4.5)", "weka.classifiers.mi.MIWrapper -P 1 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (G, C4.5)", "weka.classifiers.mi.MIWrapper -P 2 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("MIWrapper (M, C4.5)", "weka.classifiers.mi.MIWrapper -P 3 -A 3 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));

    /**
     * Wrapper with attribute space mapping
     */
    classifierEntries.add(new ClassifierEntry("MILES (C4.5)", "weka.classifiers.meta.FilteredClassifier -F \"weka.filters.unsupervised.attribute.MILESFilter -S 894.4271909999159\" -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));

    /**
     * Boosting
     */
    classifierEntries.add(new ClassifierEntry("MIBoost (C4.5)", "weka.classifiers.mi.MIBoost -R 10 -B 0 -W weka.classifiers.trees.J48 -- -C 0.25 -M 2"));
    classifierEntries.add(new ClassifierEntry("AdaBoost (MITI)", "weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.mi.MITI -- -K 5 -U -B -Ba 0.5 -M 2 -A 1 -An 1 -S 1"));
    classifierEntries.add(new ClassifierEntry("AdaBoost (MIRI)", "weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.mi.MIRI -- -K 5 -U -B -Ba 0.5 -M 2 -A 1 -An 1 -S 1"));

    /**
     * Logistic regression
     */
    classifierEntries.add(new ClassifierEntry("MILR (A)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 1"));
    classifierEntries.add(new ClassifierEntry("MILR (G)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 2"));
    classifierEntries.add(new ClassifierEntry("MILR (S)", "weka.classifiers.mi.MILR -R 1.0E-6 -A 0"));

    /**
     * Support vector machines
     */
    classifierEntries.add(new ClassifierEntry("MISVM (K1)", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
    classifierEntries.add(new ClassifierEntry("MISVM (K2)", "weka.classifiers.mi.MISVM -C 1.0 -N 0 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 2.0\""));
    classifierEntries.add(new ClassifierEntry("MISMO (K1)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 1.0\""));
    classifierEntries.add(new ClassifierEntry("MISMO (K2)", "weka.classifiers.mi.MISMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.mi.supportVector.MIPolyKernel -C 250007 -E 2.0\""));
  }

  public int numClassifiers() {
    return classifierEntries.size();
  }

  public ClassifierEntry classifierEntry(int index) {
    return classifierEntries.get(index);
  }

}
