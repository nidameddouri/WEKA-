/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    AdaBoostM2.java
 *    Copyright (C) 2011 Faculty of Sciences, El Manar, Tunisia
 *
 */

package weka.classifiers.meta;

import weka.classifiers.Classifier;
//import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.Sourcable;
//import weka.classifiers.lattices.Rules;
//import weka.classifiers.evaluation.Diversity;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

//import java.io.BufferedWriter;
//import java.io.FileWriter;
//import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/**
 <!-- globalinfo-start -->
 * Class for boosting a nominal class classifier using the Adaboost M1 method. Only nominal class problems can be tackled. Often dramatically improves performance, but sometimes overfits.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Yoav Freund, Robert E. Schapire: Experiments with a new boosting algorithm. In: Thirteenth International Conference on Machine Learning, San Francisco, 148-156, 1996.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Freund1996,
 *    address = {San Francisco},
 *    author = {Yoav Freund and Robert E. Schapire},
 *    booktitle = {Thirteenth International Conference on Machine Learning},
 *    pages = {148-156},
 *    publisher = {Morgan Kaufmann},
 *    title = {Experiments with a new boosting algorithm},
 *    year = {1996}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -P &lt;num&gt;
 *  Percentage of weight mass to base training on.
 *  (default 100, reduce to around 90 speed up)</pre>
 * 
 * <pre> -Q
 *  Use resampling for boosting.</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.J48)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.trees.DecisionStump:
 * </pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Nidà Meddouri (nida.meddouri@gmail.com)
// * @version $Revision: 5928 $ 
 */
public class AdaBoostHD 
  extends RandomizableIteratedSingleClassifierEnhancer 
  implements WeightedInstancesHandler, Sourcable, TechnicalInformationHandler {

  /** for serialization */
  static final long serialVersionUID = -2020091761919L;
  
  /** Max num iterations tried to find classifier with non-zero error. */ 
  //private static int MAX_NUM_RESAMPLING_ITERATIONS = 100;
  
  /** Array for storing the weights for the votes. */
  //protected double [] m_Betas;
  protected ArrayList <Double> m_Betas;

  /** The number of successfully generated base classifiers. */
  protected int m_NumIterationsPerformed;
  
  /** me: */
  
  /**The weight vector*/
  protected ArrayList <ArrayList <Double>> weightV;
	
  /**Définir une sutructure contenant les poids des instances à chaque itération */
  protected ArrayList <Double> WeightInst_byIteration ;
  
  /** Etude de la marge*/
  //public static ArrayList <Double> vect_edge_it ;
  
  /** Etude de la marge*/
  //public static ArrayList <Double> vect_errorTraining_it ;
  
  /** Etude de la marge*/
  //public static ArrayList <Double> vect_epsilon_it ;
  
  /** Etude de la marge*/
  //public static ArrayList <Double> vect_beta_it ;
  
  /** Vecteur contenant les prédiction de chaque classifieur faible pour tout les instances du contexte (nb_it*nb_inst)*/
  public static ArrayList <ArrayList <Integer>> prediction_byWeak_it;
  
  /** Weight Threshold. The percentage of weight mass used in training */
  protected int m_WeightThreshold = 100;

  /** Use boosting with reweighting? */
  protected boolean m_UseResampling = false;

  /** The number of classes */
  protected int m_NumClasses;
  
  /** a ZeroR model in case no model can be built from the data */
  protected Classifier m_ZeroR;
  
//  protected Diversity m_diversity;
    
  /**
	 * 
	 */  

  private  double Variance =0.001 ; 
  
	// Valeur initial de la variance des valeurs de Q	
	public double previousValuesQ = 0.0;
	
	// Valeur initial de la variance des valeurs de CC
	public double previousValuesCC = 0.0;
	
	// Valeur initial de la variance des valeurs de KP
	public double previousValueskp = 0.0;
	

	
  public static final int NoMeasureOfDiversity=0;
  public static final int DiversityMeasure_Q=1;  	
  public static final int DiversityMeasure_CC=2;  
  public static final int DiversityMeasure_kp=3;
  
  private int DiversityMeasure=DiversityMeasure_Q;
      
  public static final Tag [] TAGS_DiversityMeasure = 
  {
	  new Tag(NoMeasureOfDiversity, 		" "),	
	  new Tag(DiversityMeasure_Q, 		"Q Statistique"),	
	  new Tag(DiversityMeasure_CC, 		"Coeficient de Correlation"),
	  new Tag(DiversityMeasure_kp, 		"Pairwise Interrater Agreement"),
	  };

	public SelectedTag getDiversityMeasure() {
		return new SelectedTag(DiversityMeasure, TAGS_DiversityMeasure);
	}

	public void setDiversityMeasure(SelectedTag agregation) {
		if (agregation.getTags() == TAGS_DiversityMeasure) {
			this.DiversityMeasure = agregation.getSelectedTag().getID();
		}
	}
	
	/**
	 * 
	 */
	
  /**
   * Constructor.
   */
  public AdaBoostHD() {   
//	  m_Classifier = new weka.classifiers.trees.DecisionStump();
  }
    
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
 
    return "Class for boosting a nominal class classifier using the Adaboost "
      + "M2 method. Only nominal class problems can be tackled. Often "
      + "dramatically improves performance, but sometimes overfits.\n\n"
      + "For more information, see\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.INPROCEEDINGS);
    result.setValue(Field.AUTHOR, "Nida Meddouri, Hela Khoufi  and Mondher maddouri");
    result.setValue(Field.TITLE, "");
    result.setValue(Field.BOOKTITLE, "");
    result.setValue(Field.YEAR, "2011");
    result.setValue(Field.PAGES, "000-000");
    result.setValue(Field.PUBLISHER, " ");
    result.setValue(Field.ADDRESS, " ");
    
    return result;
  }

  /**
   * String describing default classifier.
   * 
   * @return the default classifier classname
   */
  protected String defaultClassifierString() {
    
//    return "weka.classifiers.trees.J48";
	    return "weka.classifiers.trees.DecisionStump";
  }

  /**
   * Select only instances with weights that contribute to 
   * the specified quantile of the weight distribution
   *
   * @param data the input instances
   * @param quantile the specified quantile eg 1.0 to select 
   * 100% of the weight mass
   * @return the selected instances
   */
  protected Instances selectWeightQuantile(Instances data, double quantile) { 

    int numInstances = data.numInstances();
    Instances trainData = new Instances(data, numInstances);
    double [] weights = new double [numInstances];

    double sumOfWeights = 0;
    for(int i = 0; i < numInstances; i++) {
      weights[i] = data.instance(i).weight();
      sumOfWeights += weights[i];
    }
    double weightMassToSelect = sumOfWeights * quantile;
    int [] sortedIndices = Utils.sort(weights);

    // Select the instances
    sumOfWeights = 0;
    for(int i = numInstances - 1; i >= 0; i--) {
      Instance instance = (Instance)data.instance(sortedIndices[i]).copy();
      trainData.add(instance);
      sumOfWeights += weights[sortedIndices[i]];
      if ((sumOfWeights > weightMassToSelect) && 
	  (i > 0) && 
	  (weights[sortedIndices[i]] != weights[sortedIndices[i - 1]])) {
	break;
      }
    }
    if (m_Debug) {
      System.err.println("Selected " + trainData.numInstances()
			 + " out of " + numInstances);
    }
    return trainData;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector();

    newVector.addElement(new Option(
	"\tPercentage of weight mass to base training on.\n"
	+"\t(default 100, reduce to around 90 speed up)",
	"P", 1, "-P <num>"));
    
    newVector.addElement(new Option(
	"\tUse resampling for boosting.",
	"Q", 0, "-Q"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }
    
    return newVector.elements();
  }

  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -P &lt;num&gt;
   *  Percentage of weight mass to base training on.
   *  (default 100, reduce to around 90 speed up)</pre>
   * 
   * <pre> -Q
   *  Use resampling for boosting.</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 50)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.trees.DecisionStump)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.trees.DecisionStump:
   * </pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   <!-- options-end -->
   *
   * Options after -- are passed to the designated classifier.<p>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    String thresholdString = Utils.getOption('P', options);
    if (thresholdString.length() != 0) {
      setWeightThreshold(Integer.parseInt(thresholdString));
    } else {
      setWeightThreshold(100);
    }
      
    setUseResampling(Utils.getFlag('Q', options));
    
    super.setOptions(options);
    
    Variance = this.getVariance();
    
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;
    
    result = new Vector();

    if (getUseResampling())
      result.add("-Q");

    result.add("-P");
    result.add("" + getWeightThreshold());
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    /**
     * 
     */
    
	 Double var = this.getVariance();
	 result.add(" -"+var.toString());
	 
	 /**
	  * 
	  */

    return (String[]) result.toArray(new String[result.size()]);
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String weightThresholdTipText() {
    return "Weight threshold for weight pruning.";
  }

  /**
   * Set weight threshold
   *
   * @param threshold the percentage of weight mass used for training
   */
  public void setWeightThreshold(int threshold) {

    m_WeightThreshold = threshold;
  }

  /**
   * Get the degree of weight thresholding
   *
   * @return the percentage of weight mass used for training
   */
  public int getWeightThreshold() {

    return m_WeightThreshold;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useResamplingTipText() {
    return "Whether resampling is used instead of reweighting.";
  }

  /**
   * Set resampling mode
   *
   * @param r true if resampling should be done
   */
  public void setUseResampling(boolean r) {

    m_UseResampling = r;
  }

  /**
   * Get whether resampling is turned on
   *
   * @return true if resampling output is on
   */
  public boolean getUseResampling() {

    return m_UseResampling;
  }
  
  /**
   * 
   */
  
  	public double getVariance() { 	return  Variance; }
	
	public void setVariance (double Var) { Variance = Var; }
  
  /**
   * 
   */

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // class
    result.disableAllClasses();
    result.disableAllClassDependencies();
    if (super.getCapabilities().handles(Capability.NOMINAL_CLASS))
      result.enable(Capability.NOMINAL_CLASS);
    if (super.getCapabilities().handles(Capability.BINARY_CLASS))
      result.enable(Capability.BINARY_CLASS);
    
    return result;
  }

  /**
   * Boosting method.
   *
   * @param data the training data to be used for generating the
   * boosted classifier.
   * @throws Exception if the classifier could not be built successfully
   */

  public void buildClassifier(Instances data) throws Exception {

	  //Nida
	//System.err.print("\n"+this.getClassifierSpec()+"/");  
    
	super.buildClassifier(data);

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    // only class? -> build ZeroR model
    if (data.numAttributes() == 1) {
      System.err.println(
	  "Cannot build model (only class attribute present in data!), "
	  + "using ZeroR model instead!");
      m_ZeroR = new weka.classifiers.rules.ZeroR();
      m_ZeroR.buildClassifier(data);
      return;
    }
    else {
      m_ZeroR = null;
    }
    
    m_NumClasses = data.numClasses();
    if ((!m_UseResampling) && (m_Classifier instanceof WeightedInstancesHandler)) {
      buildClassifierWithWeights(data);
    } else {
      buildClassifierUsingResampling(data);
    }
  }

  /**
   * Boosting method. Boosts using resampling
   *
   * @param data the training data to be used for generating the
   * boosted classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  protected void buildClassifierUsingResampling(Instances data) 
    throws Exception {
	  
	//System.out.print("*");
	//System.out.println(this.getClassifierSpec());

    Instances trainData, sample, training;
    double epsilon, reweight, sumProbs;
    Evaluation evaluation;
    int numInstances = data.numInstances();
    Random randomInstance = new Random(m_Seed);
    int resamplingIterations = 0;
    
    boolean highDiversity = false;

    // Initialize data
    //m_Betas = new double [m_Classifiers.length];
    m_Betas = new ArrayList <Double> ();
    m_NumIterationsPerformed = 0;
    
    // Create a copy of the data so that when the weights are diddled
    // with it doesn't mess up the weights for anyone else
    training = new Instances(data, 0, numInstances);
    sumProbs = training.sumOfWeights();
    for (int i = 0; i < training.numInstances(); i++) {
      training.instance(i).setWeight(training.instance(i).
				      weight() / sumProbs);
    }
    
    // Vecteur contenant les prédiction de chaque classifieur faible pour tout les instances du contexte
	prediction_byWeak_it = new ArrayList <ArrayList <Integer>> ();  
	// Vecteur contenant les avantages (edge) de chaque classifieur faible (associée à chaque itération)
	//vect_edge_it= new ArrayList <Double> ();
	// Vecteur contenant les erreurs d'apprentissage de chaque classifieur faible (associée à chaque itération)
	//vect_errorTraining_it= new ArrayList <Double> ();
	// Vecteur contenant les erreurs de test de chaque classifieur faible (associée à chaque itération)
	//vect_epsilon_it= new ArrayList <Double> ();

	// Vecteur contenant les erreurs de test pondérées de chaque classifieur faible (beta associée à chaque itération)
	//vect_beta_it= new ArrayList <Double> ();

	// Vecteur contenant les poids associées à chaque classifieur ( à chaque itération)
	WeightInst_byIteration = new ArrayList<Double>(training.numInstances());

	
	//INPUT: Distribution D over the N examples
	for (int i = 0; i < training.numInstances(); i++)
		training.instance(i).setWeight((double) 1/training.numInstances());

	//INITIALIZE the weight vector: w1 i,y = D(i)/(k-1) for i=1,..,N and y in Y-{yi}.
	weightV = new ArrayList <ArrayList <Double>> ();
	for(int i=0 ; i<training.numInstances() ; i++)
	{
		ArrayList<Double> weightIns = new ArrayList <Double>();
		for(int j=0;j<training.numClasses();j++)
			if(j==training.instance(i).classValue())
				weightIns.add((double) 0);
			else
                weightIns.add((double) training.instance(i).weight() / (training.numClasses() - 1));
		weightV.add(weightIns);
	}
	    
    // Do boostrap iterations
    for (m_NumIterationsPerformed = 0; highDiversity == false && m_NumIterationsPerformed < m_Classifiers.length; m_NumIterationsPerformed++) {
      if (m_Debug) {
	System.err.println("Training classifier " + (m_NumIterationsPerformed + 1));
      }

    //Calculer la somme des weights pour chaque instance du context
		Double somW;
		WeightInst_byIteration.clear();
		for(int i=0;i<training.numInstances();i++)
		{
			somW = (double) 0;
			for(int j=0;j<training.numClasses();j++)
				somW += weightV.get(i).get(j);
			WeightInst_byIteration.add(somW);
		}
		
		//Calculer les 'qweights' pour chaque instance du context
		ArrayList <ArrayList <Double>> qweightV = new ArrayList<ArrayList<Double>>();
		qweightV.clear();
		for(int i=0;i<training.numInstances();i++)
		{
			ArrayList <Double> qweightIns = new ArrayList<Double>(training.numClasses());
			for(int j=0;j<training.numClasses();j++)
				qweightIns.add((double) weightV.get(i).get(j) / WeightInst_byIteration.get(i));
			qweightV.add(qweightIns);
		}
		
		//Sommation des weight pour cette itération
		double somWit = 0;
		for(int i=0;i<training.numInstances();i++)
			somWit += WeightInst_byIteration.get(i);
		
		//Les nouvelles valeur de distribution pour les instances du contexte
		for(int i=0;i<training.numInstances();i++)
		{
			Double temp=(Double) WeightInst_byIteration.get(i) / somWit;
			training.instance(i).setWeight(temp);
		}
	
      // Select instances to train the classifier on
      if (m_WeightThreshold < 100) {
	trainData = selectWeightQuantile(training, 
					 (double)m_WeightThreshold / 100);
      } else {
	trainData = new Instances(training);
      }
      
      // Resample
      resamplingIterations = 0;
      double[] weights = new double[trainData.numInstances()];
      for (int i = 0; i < weights.length; i++) {
	weights[i] = trainData.instance(i).weight();
      }

	sample = trainData.resampleWithWeights(randomInstance, weights);

	// Build and evaluate classifier
	m_Classifiers[m_NumIterationsPerformed].buildClassifier(sample);
	evaluation = new Evaluation(data);
	evaluation.evaluateModel(m_Classifiers[m_NumIterationsPerformed],training);
      
    //Les predictions du classifieur faible courant pour chaque instance du contexte
      ArrayList <Integer> prediction_inst_byWeak = new ArrayList <Integer>();
	  prediction_inst_byWeak.clear();
      for(int i=0; i<training.numInstances();i++)
      {
    	  if(training.instance(i).classValue() == 
    		  evaluation.evaluateModelOnceAndRecordPrediction(m_Classifiers[m_NumIterationsPerformed], training.instance(i)))
    		  prediction_inst_byWeak.add(1); 
    	  else
    		  prediction_inst_byWeak.add(0);
      }
      if(m_Debug)
    	  System.out.println(" Classification OutPut: "+prediction_inst_byWeak.toString());
      
      // Calculer l'erreur d'apprentissage du classifier généré sur la totalité de l'échantillon
      double ErrorTraining = 0.0;
      //double AccuracyTraining = 0.0;
	  for(int i=0;i<training.numInstances();i++)
		  //ErrorTraining += classifieur_uni_nom(training, i,ClassifieurFaible.get(0));
		  ErrorTraining += prediction_inst_byWeak.get(i);
	  ErrorTraining /=training.numInstances();
	  ErrorTraining = 1 - ErrorTraining;
	  //vect_errorTraining_it.add(ErrorTraining);
	  evaluation.evaluateModel(m_Classifiers[m_NumIterationsPerformed], training);
	  if(m_Debug)
    	  System.out.println(" Error Training: " + ErrorTraining*100 + " Ponderated Error Training: " + evaluation.errorRate()*100 );
	  
	  //Calcul du pseudo-perte associé à chaque prédiction d'instance par le classifieur faible pour cette itération
	  ArrayList <Double> tab_pseudo_perte = new ArrayList <Double> (training.numInstances());
	  
	  //epsilon
	  epsilon = (double) 0.0; 	  
	  for(int i=0;i<training.numInstances();i++)
	  { 
		double somMultiCl=0;
		for(int j=0; j<training.numClasses(); j++)		
			if(evaluation.evaluateModelOnceAndRecordPrediction(m_Classifiers[m_NumIterationsPerformed], training.instance(i)) == j)
				somMultiCl+= (double) qweightV.get(i).get(j);// * Mclassifieur_multi_cl_nom(inst, i,ClassifieurFaible,j);

		tab_pseudo_perte.add( (double) (0.5 * training.instance(i).weight() * (1 - prediction_inst_byWeak.get(i) + somMultiCl )));
		epsilon = epsilon + tab_pseudo_perte.get(i);		
	}
	
	//Calculer epsilon
	epsilon/=training.numInstances();
	//vect_epsilon_it.add(epsilon);
	if (this.m_Debug)
		System.out.println(" The pseudo loss (epsilon) of the weak classifier: "+epsilon);
	
	/*
	 * 
	 */
	prediction_byWeak_it.add(prediction_inst_byWeak);
	
//	m_diversity.setbin_output(prediction_byWeak_it);
//	m_diversity.abcd();
	
	ArrayList <ArrayList<Double>> abcd = new ArrayList <ArrayList<Double>>();
	abcd = DiversityMeasure(prediction_byWeak_it);
	
	switch (this.DiversityMeasure)
	  	{	
  	  case NoMeasureOfDiversity: 
  		  break;
  		  
  	  case DiversityMeasure_Q:
  		  	double currentValuesQ = Q_DivMeas(abcd);	
  			if(this.Variance != 0.0 && m_NumIterationsPerformed > 2 && Math.abs(currentValuesQ - previousValuesQ) < this.Variance)
  				highDiversity = true;
  			previousValuesQ = currentValuesQ;
  			//System.out.println((m_NumIterationsPerformed+1)+ " Err: " + ErrorTraining * 100 + "Err. Pond.:" + evaluation.errorRate() + " Q: " + currentValuesQ);
  			//System.out.println((m_NumIterationsPerformed+1)+ "\t" + ErrorTraining * 100 + "\t" + evaluation.errorRate() + "\t" + currentValuesQ);
  		  break; //
  			
  	  case DiversityMeasure_CC: 
  			double currentValuesCC = CC_DivMeas(abcd);	
  			if(this.Variance != 0.0 && m_NumIterationsPerformed > 2 &&  Math.abs(currentValuesCC - previousValuesCC) < this.Variance)
  				highDiversity = true;
  			previousValuesCC = currentValuesCC;
  		  	//System.out.println((m_NumIterationsPerformed+1)+ " Err: " + ErrorTraining * 100 + "Err. Pond.:" + evaluation.errorRate() + " CC: " + currentValuesCC);
  			break; //
  		  
  	  case DiversityMeasure_kp: 
  			double currentValueskp = kp_DivMeas(abcd);	
  			if(this.Variance != 0.0 && m_NumIterationsPerformed > 2 && Math.abs(currentValueskp - previousValueskp) < this.Variance)
  				highDiversity = true;
  			previousValueskp = currentValueskp;
  		  	//System.out.println((m_NumIterationsPerformed+1)+ " Err: " + ErrorTraining * 100 + "Err. Pond.:" + evaluation.errorRate() + " kp: " + currentValueskp);
  			break; //
  	  }

	//Nida
	//System.out.println((m_NumIterationsPerformed+1)+ " Err: " + ErrorTraining * 100  +" Q: " + Q_DivMeas(abcd) + " CC: " + CC_DivMeas(abcd) + " kp: " + kp_DivMeas(abcd));
	//System.out.println((m_NumIterationsPerformed+1)+ "\t" + ErrorTraining * 100  +" \t " + Q_DivMeas(abcd) + " \t " + CC_DivMeas(abcd) + " \t " + kp_DivMeas(abcd));	
	//System.out.print("/"+(m_NumIterationsPerformed+1)+ ";" + ErrorTraining * 100  +";" + Q_DivMeas(abcd)+"/");	
	
/*	System.out.println("me "+(m_NumIterationsPerformed+1)
			+ " Err: " + m_diversity.Calcul_OccuracyClassif()  
			+" Q: " + m_diversity.Q_DivMeas() 
			+ " CC: " + m_diversity.CC_DivMeas() 
			+ " kp: " + m_diversity.kp_DivMeas()
			);*/

	/*
	 * 
	 */

	// à verifier
	//vect_edge_it.add((double) 1 - 2 * epsilon);
	
	//Calcul de la perte (beta) associée a cette iteration
	double beta = 0;
	beta = (double) epsilon / ((double) 1 - epsilon);
	//vect_beta_it.add(beta);
	if (this.m_Debug)
		System.out.println(" The error (beta) of the weak classifier: "+beta);
	
	//Mise a jour de la distribution
	for(int i=0; i<training.numInstances(); i++)
		for(int j=0; j<training.numClasses(); j++)
		{
			if(evaluation.evaluateModelOnceAndRecordPrediction(m_Classifiers[m_NumIterationsPerformed], training.instance(i)) == j)
				weightV.get(i).set(j, weightV.get(i).get(j) * Math.pow( beta , 0.5 * ((int) prediction_inst_byWeak.get(i)))) ;
			else
				weightV.get(i).set(j, weightV.get(i).get(j) * Math.pow( beta , 0.5 * ((int) 1 + prediction_inst_byWeak.get(i)))) ;
		}
	
	//Calcul du poid de classification associé à cette itération
	reweight = (double) Math.log((double)1/beta);	
	// Determine the weight to assign to this model
    //m_Betas[m_NumIterationsPerformed] = reweight;
	m_Betas.add(reweight);
    if (this.m_Debug)
		System.out.println(" The weight to assign to this classifier: " + reweight);

    }
    if (this.m_Debug)
    	System.out.println("AdaBoost.HD["+this.getClassifierSpec()+"]\t"+training.relationName()+"\tnumInstances:"+training.numInstances()+"\tNumIterationsPerformed:"+(m_NumIterationsPerformed+1));
    //System.out.println("AdaBoost.HD["+this.getClassifierSpec()+"] \t NumIterationsPerformed:"+(m_NumIterationsPerformed+1));
  }

  /**
   * Sets the weights for the next iteration.
   * 
   * @param training the training instances
   * @param reweight the reweighting factor
   * @throws Exception if something goes wrong
   */
  protected void setWeights(Instances training, double reweight) 
    throws Exception {

    double oldSumOfWeights, newSumOfWeights;

    oldSumOfWeights = training.sumOfWeights();
    Enumeration enu = training.enumerateInstances();
    while (enu.hasMoreElements()) 
    {
    	Instance instance = (Instance) enu.nextElement();
    	if (!Utils.eq(m_Classifiers[m_NumIterationsPerformed].classifyInstance(instance),  instance.classValue()))
    		instance.setWeight(instance.weight() * reweight);
    }
    
    // Renormalize weights
    newSumOfWeights = training.sumOfWeights();
    enu = training.enumerateInstances();
    while (enu.hasMoreElements()) 
    {
    	Instance instance = (Instance) enu.nextElement();
    	instance.setWeight(instance.weight() * oldSumOfWeights / newSumOfWeights);
    }
  }

  /**
   * Boosting method. Boosts any classifier that can handle weighted
   * instances.
   *
   * @param data the training data to be used for generating the
   * boosted classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  protected void buildClassifierWithWeights(Instances data) 
    throws Exception {

	//System.out.print("*");
	//System.out.println(this.getClassifierSpec());
	
    Instances trainData, training;
    double epsilon; // Pseudo-losses
    double reweight;
    //double oldSumOfWeights, newSumOfWeights;
    Evaluation evaluation;
    int numInstances = data.numInstances();
    Random randomInstance = new Random(m_Seed);
    
    boolean highDiversity = false;

    // Initialize data
    //m_Betas = new double [m_Classifiers.length];
    //each Beta is calculated from epsilon: Beta = epsilon/(1-epsilon)
    m_Betas = new ArrayList <Double> ();
    m_NumIterationsPerformed = 0;

    // Create a copy of the data so that when the weights are diddled
    // with it doesn't mess up the weights for anyone else
    //INPUT: sequence of N examples <(x1,y1),...(xn,yn)> with label yi in Y={1,..,k}
    training = new Instances(data, 0, numInstances);
    
	// Vecteur contenant les prédiction de chaque classifieur faible pour tout les instances du contexte
	prediction_byWeak_it = new ArrayList <ArrayList <Integer>> ();  
	// Vecteur contenant les avantages (edge) de chaque classifieur faible (associée à chaque itération)
	//vect_edge_it= new ArrayList <Double> ();
	// Vecteur contenant les erreurs d'apprentissage de chaque classifieur faible (associée à chaque itération)
	//vect_errorTraining_it= new ArrayList <Double> ();
	// Vecteur contenant les erreurs de test de chaque classifieur faible (associée à chaque itération)
	//vect_epsilon_it= new ArrayList <Double> ();

	// Vecteur contenant les erreurs de test pondérées de chaque classifieur faible (beta associée à chaque itération)
	//vect_beta_it= new ArrayList <Double> ();

	// Vecteur contenant les poids associées à chaque classifieur ( à chaque itération)
	WeightInst_byIteration = new ArrayList<Double>(training.numInstances());

	
	//INPUT: Distribution D over the N examples
	for (int i = 0; i < training.numInstances(); i++)
		training.instance(i).setWeight((double) 1/training.numInstances());

	//INITIALIZE the weight vector: w1 i,y = D(i)/(k-1) for i=1,..,N and y in Y-{yi}.
	weightV = new ArrayList <ArrayList <Double>> ();
	for(int i=0 ; i<training.numInstances() ; i++)
	{
		ArrayList<Double> weightIns = new ArrayList <Double>();
		for(int j=0;j<training.numClasses();j++)
			if(j==training.instance(i).classValue())
				weightIns.add((double) 0);
			else
                weightIns.add((double) training.instance(i).weight() / (training.numClasses() - 1));
		weightV.add(weightIns);
	}
	
	/*
     * Do boostrap iterations 
     */
	
    for (m_NumIterationsPerformed = 0; highDiversity == false && m_NumIterationsPerformed < m_Classifiers.length ; m_NumIterationsPerformed++) 
    {    	
    	if (m_Debug)  
    		System.err.println("Training classifier " + (m_NumIterationsPerformed + 1));
    	
		//Calculer la somme des weights pour chaque instance du context
		Double somW;
		WeightInst_byIteration.clear();
		for(int i=0;i<training.numInstances();i++)
		{
			somW = (double) 0;
			for(int j=0;j<training.numClasses();j++)
				somW += weightV.get(i).get(j);
			WeightInst_byIteration.add(somW);
		}
		
		//Calculer les 'qweights' pour chaque instance du context
		ArrayList <ArrayList <Double>> qweightV = new ArrayList<ArrayList<Double>>();
		qweightV.clear();
		for(int i=0;i<training.numInstances();i++)
		{
			ArrayList <Double> qweightIns = new ArrayList<Double>(training.numClasses());
			for(int j=0;j<training.numClasses();j++)
				qweightIns.add((double) weightV.get(i).get(j) / WeightInst_byIteration.get(i));
			qweightV.add(qweightIns);
		}
		
		//Sommation des weight pour cette itération
		double somWit = 0;
		for(int i=0;i<training.numInstances();i++)
			somWit += WeightInst_byIteration.get(i);
		
		//Les nouvelles valeur de distribution pour les instances du contexte
		for(int i=0;i<training.numInstances();i++)
		{
			Double temp=(Double) WeightInst_byIteration.get(i) / somWit;
			training.instance(i).setWeight(temp);
		}
 	
      // Select instances to train the classifier on
      //Weight Threshold. The percentage of weight mass used in training
      if (m_WeightThreshold < 100) 
    	  trainData = selectWeightQuantile(training, (double)m_WeightThreshold / 100);
      else
    	  trainData = new Instances(training, 0, numInstances);
      
      // Build the classifier
      // Call WeakLearn, providing it with the distrubution Dt and label weighting function qt; get back a hypothesis ht: X*Y->[0,1]
      if (m_Classifiers[m_NumIterationsPerformed] instanceof Randomizable)
    	  ((Randomizable) m_Classifiers[m_NumIterationsPerformed]).setSeed(randomInstance.nextInt());
      m_Classifiers[m_NumIterationsPerformed].buildClassifier(trainData);

      // Evaluate the classifier
      evaluation = new Evaluation(data);
      
      //Les predictions du classifieur faible courant pour chaque instance du contexte
      ArrayList <Integer> prediction_inst_byWeak = new ArrayList <Integer>();
	  prediction_inst_byWeak.clear();
      for(int i=0; i<training.numInstances();i++)
      {
    	  if(training.instance(i).classValue() == 
    		  evaluation.evaluateModelOnceAndRecordPrediction(m_Classifiers[m_NumIterationsPerformed], training.instance(i)))
    		  prediction_inst_byWeak.add(1); 
    	  else
    		  prediction_inst_byWeak.add(0);
      }
      if(m_Debug)
    	  System.out.println(" Classification OutPut: "+prediction_inst_byWeak.toString());
      
      // Calculer l'erreur d'apprentissage du classifier généré sur la totalité de l'échantillon
	  double ErrorTraining = 0.0;
	  for(int i=0;i<training.numInstances();i++)
		  //ErrorTraining += classifieur_uni_nom(training, i,ClassifieurFaible.get(0));
		  ErrorTraining += prediction_inst_byWeak.get(i);
	  ErrorTraining/=training.numInstances();
	  ErrorTraining = 1 - ErrorTraining;
	  //vect_errorTraining_it.add(ErrorTraining);
	  evaluation.evaluateModel(m_Classifiers[m_NumIterationsPerformed], training);
	  if(m_Debug)
    	  System.out.println(" Error Training: " + ErrorTraining*100 + " Ponderated Error Training: " + evaluation.errorRate()*100 );
	  
	  //Calcul du pseudo-perte associé à chaque prédiction d'instance par le classifieur faible pour cette itération
	  ArrayList <Double> tab_pseudo_perte = new ArrayList <Double> (training.numInstances());
	  
	  //epsilon
	  epsilon = (double) 0.0; 	  
	  for(int i=0;i<training.numInstances();i++)
	  { 
		double somMultiCl=0;
		for(int j=0; j<training.numClasses(); j++)		
			if(evaluation.evaluateModelOnceAndRecordPrediction(m_Classifiers[m_NumIterationsPerformed], training.instance(i)) == j)
				somMultiCl+= (double) qweightV.get(i).get(j);// * Mclassifieur_multi_cl_nom(inst, i,ClassifieurFaible,j);

		tab_pseudo_perte.add( (double) (0.5 * training.instance(i).weight() * (1 - prediction_inst_byWeak.get(i) + somMultiCl )));
		epsilon = epsilon + tab_pseudo_perte.get(i);		
	}
	
	//Calculer epsilon
	epsilon/=training.numInstances();
	//vect_epsilon_it.add(epsilon);
	if (this.m_Debug)
		System.out.println(" The pseudo loss (epsilon) of the weak classifier: "+epsilon);
	
	/*
	 * 
	 */
	prediction_byWeak_it.add(prediction_inst_byWeak);

	ArrayList <ArrayList<Double>> abcd = new ArrayList <ArrayList<Double>>();
	abcd = DiversityMeasure(prediction_byWeak_it);
	
	switch (this.DiversityMeasure)
	  	{	
  	  case NoMeasureOfDiversity: 
  		  break;
  		  
  	  case DiversityMeasure_Q:
  		  double currentValuesQ = Q_DivMeas(abcd);	
  		  if(this.Variance != 0.0 && m_NumIterationsPerformed > 2 && Math.abs(currentValuesQ - previousValuesQ) < this.Variance)
  			  highDiversity = true;
  		  previousValuesQ = currentValuesQ;
  		  //System.out.println((m_NumIterationsPerformed+1)+ " Err: " + ErrorTraining * 100 + "Err. Pond.:" + evaluation.errorRate() + " Q: " + currentValuesQ);
  		  //System.out.println((m_NumIterationsPerformed+1)+ "\t" + ErrorTraining * 100 + "\t" + evaluation.errorRate() + "\t" + currentValuesQ);
		  
  		  break; //
  			
  	  case DiversityMeasure_CC: 
  			 double currentValuesCC = CC_DivMeas(abcd);	
  			if(this.Variance != 0.0 && m_NumIterationsPerformed > 2 &&  Math.abs(currentValuesCC - previousValuesCC) < this.Variance)
  				highDiversity = true;
  			previousValuesCC = currentValuesCC;
  		  	//System.out.println((m_NumIterationsPerformed+1)+ " Err: " + ErrorTraining * 100 + "Err. Pond.:" + evaluation.errorRate()+  " CC: " + currentValuesCC);
  			break; //
  		  
  	  case DiversityMeasure_kp: 
  			double currentValueskp = kp_DivMeas(abcd);	
  			if(this.Variance != 0.0 && m_NumIterationsPerformed > 2 && Math.abs(currentValueskp - previousValueskp) < this.Variance)
  				highDiversity = true;
  			previousValueskp = currentValueskp;
  		  	//System.out.println((m_NumIterationsPerformed+1)+ " Err: " + ErrorTraining * 100 + "Err. Pond.:" + evaluation.errorRate() +  " kp: " + currentValueskp);  		  	
  			break; //
  	  }
	
 
	//Nida
	//System.out.println((m_NumIterationsPerformed+1)+ " Err: " + ErrorTraining * 100  +" Q: " + Q_DivMeas(abcd) + " CC: " + CC_DivMeas(abcd) + " kp: " + kp_DivMeas(abcd));
	//System.out.println((m_NumIterationsPerformed+1)+ "\t" + ErrorTraining * 100  +" \t " + Q_DivMeas(abcd) + " \t " + CC_DivMeas(abcd) + " \t " + kp_DivMeas(abcd));	
	//System.out.print("/"+(m_NumIterationsPerformed+1)+ ";" + ErrorTraining * 100  +";" + Q_DivMeas(abcd)+"/");	


	// à verifier
	//vect_edge_it.add((double) 1 - 2 * epsilon);
	
	//Calcul de la perte (beta) associée a cette iteration
	double beta = 0;
	beta = (double) epsilon / ((double) 1 - epsilon);
	//vect_beta_it.add(beta);
	if (this.m_Debug)
		System.out.println(" The error (beta) of the weak classifier: "+beta);
	
	//Mise a jour de la distribution
	for(int i=0; i<training.numInstances(); i++)
		for(int j=0; j<training.numClasses(); j++)
		{
			if(evaluation.evaluateModelOnceAndRecordPrediction(m_Classifiers[m_NumIterationsPerformed], training.instance(i)) == j)
				weightV.get(i).set(j, weightV.get(i).get(j) * Math.pow( beta , 0.5 * ((int) prediction_inst_byWeak.get(i)))) ;
			else
				weightV.get(i).set(j, weightV.get(i).get(j) * Math.pow( beta , 0.5 * ((int) 1 + prediction_inst_byWeak.get(i)))) ;
		}
	
	//Calcul du poid de classification associé à cette itération
	reweight = (double) Math.log((double)1/beta);	
	// Determine the weight to assign to this model
    //m_Betas[m_NumIterationsPerformed] = reweight;
	m_Betas.add(reweight);
    if (this.m_Debug)
		System.out.println(" The weight to assign to this classifier: " + reweight);
      
      
    }// End for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_Classifiers.length; m_NumIterationsPerformed++)
    if (this.m_Debug)
    	System.out.println(this.getClassifierSpec()+"\t"+training.relationName()+"\t"+training.numInstances()+"\t"+(m_NumIterationsPerformed+1));
  
  }// End buildClassifierWithWeights(Instances data) 
  
  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @throws Exception if instance could not be classified
   * successfully
   */
  public double [] distributionForInstance(Instance instance) 
    throws Exception {
      
    // default model?
    if (m_ZeroR != null) {
      return m_ZeroR.distributionForInstance(instance);
    }
    
    if (m_NumIterationsPerformed == 0) {
      throw new Exception("No model built");
    }
    double [] sums = new double [instance.numClasses()]; 
    
    if (m_NumIterationsPerformed == 1) {
      return m_Classifiers[0].distributionForInstance(instance);
    } else {
      for (int i = 0; i < m_NumIterationsPerformed; i++) {
	sums[(int)m_Classifiers[i].classifyInstance(instance)] += /*m_Betas[i]*/m_Betas.get(i);
      }
      return Utils.logs2probs(sums);
    }
  }

  /**
   * Returns the boosted model as Java source code.
   *
   * @param className the classname of the generated class
   * @return the tree as Java source code
   * @throws Exception if something goes wrong
   */
  public String toSource(String className) throws Exception {

    if (m_NumIterationsPerformed == 0) {
      throw new Exception("No model built yet");
    }
    if (!(m_Classifiers[0] instanceof Sourcable)) {
      throw new Exception("Base learner " + m_Classifier.getClass().getName()
			  + " is not Sourcable");
    }

    StringBuffer text = new StringBuffer("class ");
    text.append(className).append(" {\n\n");

    text.append("  public static double classify(Object[] i) {\n");

    if (m_NumIterationsPerformed == 1) {
      text.append("    return " + className + "_0.classify(i);\n");
    } else {
      text.append("    double [] sums = new double [" + m_NumClasses + "];\n");
      for (int i = 0; i < m_NumIterationsPerformed; i++) {
	text.append("    sums[(int) " + className + '_' + i 
		    + ".classify(i)] += " + /*m_Betas[i]*/ m_Betas.get(i) + ";\n");
      }
      text.append("    double maxV = sums[0];\n" +
		  "    int maxI = 0;\n"+
		  "    for (int j = 1; j < " + m_NumClasses + "; j++) {\n"+
		  "      if (sums[j] > maxV) { maxV = sums[j]; maxI = j; }\n"+
		  "    }\n    return (double) maxI;\n");
    }
    text.append("  }\n}\n");

    for (int i = 0; i < m_Classifiers.length; i++) {
	text.append(((Sourcable)m_Classifiers[i])
		    .toSource(className + '_' + i));
    }
    return text.toString();
  }

  /**
   * Returns description of the boosted classifier.
   *
   * @return description of the boosted classifier as a string
   */
  public String toString() {
    
    // only ZeroR model?
    if (m_ZeroR != null) {
      StringBuffer buf = new StringBuffer();
      buf.append(this.getClass().getName().replaceAll(".*\\.", "") + "\n");
      buf.append(this.getClass().getName().replaceAll(".*\\.", "").replaceAll(".", "=") + "\n\n");
      buf.append("Warning: No model could be built, hence ZeroR model is used:\n\n");
      buf.append(m_ZeroR.toString());
      return buf.toString();
    }
    
    StringBuffer text = new StringBuffer();
    
    if (m_NumIterationsPerformed == 0) {
      text.append("AdaBoost.HD: No model built yet.\n");
    } else if (m_NumIterationsPerformed == 1) {
      text.append("AdaBoost.HD: No boosting possible, one classifier used!\n");
      text.append(m_Classifiers[0].toString() + "\n");
    } else {
      text.append("AdaBoost.HD: Base classifiers and their weights: \n\n");
      for (int i = 0; i < m_NumIterationsPerformed ; i++) {
    	  text.append(m_Classifiers[i].toString() + "\n\n");
    	  text.append("Classifier " + i + "\t Weight: " + Utils.roundDouble(m_Betas.get(i), 2) + "\n");
      }
      text.append("Number of performed Iterations: " + m_NumIterationsPerformed + "\n");
    	text.append("AdaBoost.HD: Possible boosting, many classifier used!\n");
    }
    
    return text.toString();
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 2020 $");
  }

  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
    runClassifier(new AdaBoostHD(), argv);
  }

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  public final ArrayList <ArrayList <Double>> DiversityMeasure(ArrayList <ArrayList <Integer>> All_result_classif)
  {
	  ArrayList <ArrayList <Double>> abcd = new ArrayList <ArrayList <Double>>();
	  for (int i=0; i<(int) All_result_classif.size();i++)
		  for(int j=i+1; j<(int) All_result_classif.size();j++)
		  {
			  ArrayList <Double> temp_abcd = new ArrayList <Double>();
			  temp_abcd = pairwiseABCD(All_result_classif.get(i), All_result_classif.get(j));
			  abcd.add(temp_abcd);
		  }
	  return abcd;	  
  }
  
  public final ArrayList <Double> Calcul_OccuracyClassif(ArrayList <ArrayList <Integer>> All_result_classif)
  {
	  ArrayList <Double> each_occuracy_classif = new ArrayList <Double>();
	  int Compteur_Classif = 0;
	  while(Compteur_Classif < All_result_classif.get(0).size())
	  {
		  Integer Correctly_Classif = 0;
		  for(int i=0; i<(int) All_result_classif.size();i++)
		  {
			  if (All_result_classif.get(i).get(Compteur_Classif)==1)
				  Correctly_Classif ++;			  
		  }
		  Double occuracy = (Double) (Correctly_Classif / (double) All_result_classif.get(0).size());
		  each_occuracy_classif.add((Double)occuracy*100);
		  Compteur_Classif++;		  
	  }
	  
	  for(int i=0; i<each_occuracy_classif.size(); i++)
		  System.out.println(i+" * "+each_occuracy_classif.get(i));

	  ArrayList <Double> occuracy_classif = new ArrayList <Double>();
	  for (int i=0; i<(int) each_occuracy_classif.size();i++)
		  for(int j=i+1; j<(int) each_occuracy_classif.size();j++)
		  {
			  Double temp_occuracy_classif = (Double) ((each_occuracy_classif.get(i) + each_occuracy_classif.get(j))/2);			  
			  occuracy_classif.add(temp_occuracy_classif);
		  }
	  
//	  for(int i=0; i<occuracy_classif.size(); i++)
//		  System.out.println(i+" *** "+occuracy_classif.get(i));
	  
	  return occuracy_classif;	  
  }
  
  
  /**
   * Returns value of Q statistic if class is nominal.
   *
   * @return the value of the Q statistic
   */

  public final Double Q_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumQ2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumQ2+=Q2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumQ2/abcd.size());			  
  }

  public final Double Q2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double ad = (Double) abcdIJ.get(0) * abcdIJ.get(3);
	  Double bc = (Double) abcdIJ.get(1) * abcdIJ.get(2);
	  Double tempQ2;
	  if(ad==0.00000000 && bc==0.00000000)
		  tempQ2 = new Double(0);
	  else
	  	  tempQ2 = (Double)(ad-bc)/(ad+bc);

	  return tempQ2;
  }
  
  /**
   * Returns value of the Correlation coefficient if class is nominal.
   *
   * @return the value of the Correlation coefficient
   */

  public final Double CC_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumCC2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumCC2+=CC2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumCC2/abcd.size());			  
  }

  public final Double CC2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempCC2;
	  Double ad = (Double) abcdIJ.get(0) * abcdIJ.get(3);
	  Double bc = (Double) abcdIJ.get(1) * abcdIJ.get(2);
	  Double aPLUSb = (Double) abcdIJ.get(0) + abcdIJ.get(1);
	  Double cPLUSd = (Double) abcdIJ.get(2) + abcdIJ.get(3);
	  Double aPLUSc = (Double) abcdIJ.get(0) + abcdIJ.get(2);
	  Double bPLUSd = (Double) abcdIJ.get(1) + abcdIJ.get(3);
	  Double multiplication = (Double) aPLUSb * cPLUSd * aPLUSc * bPLUSd;
	  if(multiplication==0.00000000)
		  tempCC2 = new Double(0);
	  else
		  tempCC2 = (Double) (ad-bc)/Math.sqrt(multiplication);
//	  System.out.println(multiplication+"  --  "+ Math.sqrt(multiplication) +"  --  "+ (Math.sqrt(multiplication)*Math.sqrt(multiplication)) );
	  return tempCC2;
  }

  /**
   * Returns value of Disagreement if class is nominal.
   *
   * @return the value of the Disagreement
   */

  public final Double Disagreement_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumDisagreement2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumDisagreement2+=CC2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumDisagreement2/abcd.size());			  
  }

  public final Double Disagreement2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempDisagreement2;
	  tempDisagreement2 = (Double) abcdIJ.get(1)+abcdIJ.get(2);
	  return tempDisagreement2;
  }
  /**
   * Returns value of Double-fault if class is nominal.
   *
   * @return the value of the Double-fault
   */

  public final Double DoubleFault_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  int cptDF=0;
	  Double SumDF2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumDF2+=abcd.get(i).get(3);
		  cptDF++;			  
	  }
	  return ((Double) SumDF2/cptDF);			  
  }

  /**
   * Returns value of Kohavi-Wolpert variance if class is nominal.
   *
   * @return the value of the Kohavi-Wolpert variance
   */

  public final Double KW_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumKW2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumKW2+=KW2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumKW2/abcd.size());			  
  }

  public final Double KW2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempKW2;
	  tempKW2 = (Double) (abcdIJ.get(1)+abcdIJ.get(2))/4 ;
	  return tempKW2;
  }

  /**
   * Returns value of the kappa if class is nominal.
   *
   * @return the value of the kappa
   */

  public final Double kappa_DivMeas(ArrayList <ArrayList <Double>> abcd, ArrayList <Double> occuracy_classif) {	  	  
	  Double Sumkappa=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
//		  System.out.print("occuracy_classif: "+occuracy_classif.get(i));
		  Sumkappa+=kappa2_DivMeas(abcd.get(i),occuracy_classif.get(i));
	  }
	  return ((Double) Sumkappa/abcd.size());			  
  }

  public final Double kappa2_DivMeas(ArrayList <Double> abcdIJ, Double occuracy_classif_IJ)
  {
	  Double tempkappa2;
	  Double bPLUSc = (Double) abcdIJ.get(1) + abcdIJ.get(2);
//	  System.out.println(bPLUSc);
	  
	  Double multiplication = (Double) (2 * occuracy_classif_IJ * (1 - occuracy_classif_IJ));
//	  System.out.print("    multiplication: "+multiplication);
	  if(multiplication==0.00000000)
		  tempkappa2 = new Double(1);
	  else
		  tempkappa2 = (Double) (1 - ((bPLUSc)/multiplication));
//	  System.out.println("     kappa: "+tempkappa2);
	  return tempkappa2;
  }
  
  /**
   * Returns value of the Measurement of interrater agreement if class is nominal.
   *
   * @return the value of the Measurement of interrater agreement
   */

  public final Double kp_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double Sumkp2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  Sumkp2+=kp2_DivMeas(abcd.get(i));
	  }
	  return ((Double) Sumkp2/abcd.size());			  
  }

  public final Double kp2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempkp2;
	  Double ad = (Double) abcdIJ.get(0) * abcdIJ.get(3);
	  Double bc = (Double) abcdIJ.get(1) * abcdIJ.get(2);
	  Double aPLUSb = (Double) abcdIJ.get(0) + abcdIJ.get(1);
	  Double cPLUSd = (Double) abcdIJ.get(2) + abcdIJ.get(3);
	  Double aPLUSc = (Double) abcdIJ.get(0) + abcdIJ.get(2);
	  Double bPLUSd = (Double) abcdIJ.get(1) + abcdIJ.get(3);
	  Double multiplication = (Double) aPLUSb * cPLUSd * aPLUSc * bPLUSd;
	  if(multiplication==0.00000000)
		  tempkp2 = new Double(0);
	  else
		  tempkp2 = (Double) (2*(ad-bc))/multiplication;
	  return tempkp2;
  }
  /**
   * Returns value of Entropy if class is nominal.
   *
   * @return the value of the Entropy
   */

  public final Double Entropy_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumEntropy2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumEntropy2+=Entropy2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumEntropy2/abcd.size());			  
  }

  public final Double Entropy2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempEntropy2 = (Double) (abcdIJ.get(1)+abcdIJ.get(2))/2;
	  return tempEntropy2;
  }
  
  /**
   * Returns value of the Difficulty if class is nominal.
   *
   * @return the value of the Difficulty
   */

  public final Double Difficulty_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumDifficulty2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumDifficulty2+=Difficulty2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumDifficulty2/abcd.size());			  
  }

  public final Double Difficulty2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempDifficulty2;
	  Double ad = (Double) abcdIJ.get(0) * abcdIJ.get(3);
	  Double aPLUSd = (Double) abcdIJ.get(0) + abcdIJ.get(3);
	  Double bPLUSc = (Double) abcdIJ.get(1) + abcdIJ.get(2);
	  tempDifficulty2 = (Double) (ad+((aPLUSd*bPLUSc)/4));
	  return tempDifficulty2;
  }

  
  
  /**
   * Returns value of the Generalized diversity if class is nominal.
   *
   * @return the value of the Generalized diversity
   */

  public final Double GD_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumGD2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumGD2+=GD2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumGD2/abcd.size());			  
  }

  public final Double GD2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempGD2;
	  Double bPLUSc = (Double) abcdIJ.get(1) + abcdIJ.get(2);
	  Double aPLUSd = (Double) abcdIJ.get(0) + abcdIJ.get(3);
	  Double quotion = (Double) (1 - aPLUSd);
	  if(quotion == 0.0000000)
		  tempGD2 = new Double (0);
	  else
		  tempGD2 = (Double) (bPLUSc/quotion);
	  return tempGD2;
  }
  
  
  /**
   * Returns value of the Coincident Failure diversity if class is nominal.
   *
   * @return the value of the Coincident Failure diversity
   */

  public final Double CFD_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumCFD2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumCFD2+=CFD2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumCFD2/abcd.size());			  
  }

  public final Double CFD2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempCFD2;
	  Double bPLUSc = (Double) abcdIJ.get(1) + abcdIJ.get(2);
	  Double quotion = (Double) (1 - abcdIJ.get(0));
	  if(quotion == 0.0000000)
		  tempCFD2 = new Double (0);
	  else
		  tempCFD2 = (Double) (bPLUSc/quotion);
	  return tempCFD2;
  }  

  public final ArrayList <Double> pairwiseABCD(ArrayList <Integer> OPbinCLASS1, ArrayList <Integer> OPbinCLASS2)
  {
	  Integer a=0, b=0, c=0, d=0;
	  for(int i =0; i<(int) OPbinCLASS1.size();i++)
	  {
		  if (OPbinCLASS1.get(i)==OPbinCLASS2.get(i))
			  if(OPbinCLASS1.get(i)== 1)
				  a++;
			  else 
				  d++;
		  else 
			  if(OPbinCLASS1.get(i)== 1) 
				  b++; 
			  else 
				  c++;
	  }
	  ArrayList <Double> abcd = new ArrayList <Double>();
	  abcd.add((double) a/OPbinCLASS1.size());
	  abcd.add((double) b/OPbinCLASS1.size());
	  abcd.add((double) c/OPbinCLASS1.size());
	  abcd.add((double) d/OPbinCLASS1.size());
	  
	  return abcd;
	  
  }
  


  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
}
