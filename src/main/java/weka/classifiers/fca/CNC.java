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
 *    CNC.java
 *    Copyright (C) 2010 Research Unit on Programming, Algorithmics and Heuristics - URPAH,
 *	  Faculty of Science of Tunis - FST,
 *	  Tunis - El Manar University,
 *	  Campus Universitaire EL Manar, 1060, Tunis, Tunisia.
 *
 */

package weka.classifiers.fca;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;

import weka.attributeSelection.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.classifiers.Evaluation;
import weka.classifiers.IterativeClassifier;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.UpdateableClassifier;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Reorder;

import java.lang.*;
import java.util.Calendar;
import java.util.Timer;
import java.util.Enumeration;
import java.util.GregorianCalendar;
import java.util.Vector;

//import com.sun.corba.se.impl.javax.rmi.CORBA.Util;

//import JFlex.Out;

/**
<!-- globalinfo-start -->
* Class for building and using a Classifier Nominal Concept. 
* Usually used in conjunction with a boosting/bagging algorithm. 
* [not sure] Does regression (based on mean-squared error) or classification (based on entropy). 
* [not sure] Missing is treated as a separate value.
* <p/>
<!-- globalinfo-end -->
<!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;article{,
 *    author = {Nida Meddouri et Mondher Maddouri},
 *    journal = {},
 *    pages = {},
 *    title = {},
 *    volume = {},
 *    year = 2011}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
*
* Typical usage: <p>
* <code>java weka.classifiers.meta.LogitBoost -I 100 -W weka.classifiers.fca.CNC 
* -t training_data </code><p>
* 
<!-- options-start -->
* Valid options are: <p/>
* 
* <pre> -D
*  If set, classifier is run in debug mode and
*  may output additional info to the console</pre>
* 
<!-- options-end -->
* 
* @author Nida Meddouri (nida.meddouri@gmail.com & nida.meddouri@islaib.rnu.tn)
* @version $Revision: 230783 $
*/

//public class CNC extends Classifier 
//implements UpdateableClassifier, OptionHandler, TechnicalInformationHandler 


public class CNC 
extends AbstractClassifier 
implements UpdateableClassifier, TechnicalInformationHandler, WeightedInstancesHandler, Sourcable
{	
	/** for serialization */
	static final long serialVersionUID = -2011091561919L;
	
	//Calendar calendar = Calendar.getInstance();
	//SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS");
	Calendar calendar;
	SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS");
	//System.err.println(sdf.format(calendar.getTime()));
	
	/** indicate that CNC is running*/	
	protected int m_CNC = 0;
	
	/** The attribute used for classification. DecisionStump.java*/
	protected int m_AttIndex;
	  
	/** The distribution of class values or the means in each subset. DecisionStump.java*/
	protected double[][] m_Distribution;
	
	/** The instances used for training. */
	protected Instances m_Instances;
	
	  /** The filter used to get rid of missing values. */
	protected ReplaceMissingValues m_Missing = new ReplaceMissingValues();
	  
	/** a ZeroR model in case no model can be built from the data */
	protected Classifier m_ZeroR;	
	
	 /** the coefficient to use for each measure */
	private int m_numcoeff;
    
    // Une structure contenant les régles uniques
    public static ArrayList <Classification_Rule> m_classifierNC ;
    
    protected static InfoGainAttributeEval  m_InfoGainAttributeEval =  new InfoGainAttributeEval();
    protected static InformationMutuelle m_InformationMutuelle = new InformationMutuelle();
    protected static HRatio m_HRatio = new HRatio();
    protected static GainRatioAttributeEval m_GainRatioAttributeEval = new GainRatioAttributeEval();
    protected static OneRAttributeEval      m_OneRAttributeEval      = new OneRAttributeEval();
    protected static CorrelationAttributeEval  m_CorrelationAttributeEval = new CorrelationAttributeEval();
    protected static SymmetricalUncertAttributeEval  m_SymmetricalUncertAttributeEval = new SymmetricalUncertAttributeEval();
    protected static ReliefFAttributeEval m_ReliefFAttributeEval = new ReliefFAttributeEval();
    protected static PrincipalComponents m_PrincipalComponents = new PrincipalComponents();
    // protected static ChiSquaredAttributeEval m_ChiSquaredAttributeEval = new ChiSquaredAttributeEval();
    protected static ClassifierAttributeEval m_ClassifierAttributeEval = new ClassifierAttributeEval();
    
  
	    
    /**
	 * L'apprentissage du concept nominal
	 */
    
	public static final int CONCEPT_LEARNING_FMAN = 1;  // Default: Fermeture de Meilleur Attribut Nominal
    
    private int NominalConceptLearning = CONCEPT_LEARNING_FMAN;
    
    public static final Tag [] TAGS_NominalConceptLearning = {
    	new Tag(CONCEPT_LEARNING_FMAN, "Closure of best nominal attribut"),
    	};
    
    public SelectedTag getConceptLearning() {	
    	return new SelectedTag(NominalConceptLearning, TAGS_NominalConceptLearning);	
    	}
    
    public void setConceptLearning(SelectedTag agregation) {
    	if (agregation.getTags() == TAGS_NominalConceptLearning)
    		this.NominalConceptLearning = agregation.getSelectedTag().getID();
    	}
    
    /**
     * Fermeture du Meilleur Attribut Nominal : choix de(s) valeur(s) nominale(s)
     */
    
    public static final int FMAN_GAIN_INFO_BV = 1;	// Default: La valeur la plus pertinente (support) de l'attribut qui maximise le Gain Informationel
    public static final int FMAN_GAIN_INFO_BA = 2;	// !!!!
    public static final int FMAN_GAIN_INFO_MV = 3;	// Les valeurs nominales de l'attribut qui maximise le Gain Informationel
    public static final int FMAN_GAIN_RATIO = 4;	// Les valeurs nominales de l'attribut qui maximise LE GAIN RATIO
    public static final int FMAN_ONE_R = 5;	// Les valeurs nominales de l'attribut qui maximise LE ONE R
    public static final int FMAN_Correlation_Att_Eval = 6; // les valeurs nominales qui  maximise le correlation attribut eval
    public static final int FMAN_Symmetrical = 7; // les valeurs nominales qui maximise de symmetrical
    public static final int FMAN_ClassifierAttributeEval = 8; // les valeurs nominales qui maximise le classifier attribute
    public static final int FMAN_ReliefFAttributeEval =9;
    public static final int FMAN_PrincipalComponents =10;
    public static final int FMAN_InformationMutuelle =11;
    public static final int FMAN_MeanCORR_RATIO =12;
    public static final int FMAN_HRATIO =13;
    private int FMANmeasure = FMAN_GAIN_INFO_BV;
    
    public static final Tag [] TAGS_FMANmeasure = {
		new Tag(FMAN_GAIN_INFO_BV, "Info. Gain & Best Value"),
		new Tag(FMAN_GAIN_INFO_BA, "Info. Gain & Best Attributt"),
		new Tag(FMAN_GAIN_INFO_MV, "Info. Gain & Multi Values "),
		new Tag(FMAN_GAIN_RATIO, "Gain Ratio & Best Value"),
		new Tag(FMAN_ONE_R, "One R & Best Value"),
		new Tag(FMAN_Correlation_Att_Eval,"Correlation & Best Value"),
		new Tag(FMAN_Symmetrical,"SymmetricalUncertAtt & Best Value"),
		new Tag(FMAN_ClassifierAttributeEval,"ClassifierAttributeEval & Best Value"),
		new Tag(FMAN_ReliefFAttributeEval,"ReliefFAttEval & Best Value"),
		new Tag(FMAN_PrincipalComponents,"PrincipalComponents & Best Value"),
		new Tag(FMAN_InformationMutuelle,"InformationMutuelle & Best Value"),
		new Tag(FMAN_HRATIO,"H-RATIO & Best Value"),
        new Tag(FMAN_MeanCORR_RATIO,"MoyenneCorr&Ratio & Best Value")};
    
    public SelectedTag getFMAN_Measure() {
    	return new SelectedTag(FMANmeasure, TAGS_FMANmeasure);
		
		}

	public void setFMAN_Measure(SelectedTag agregation) {
		if (agregation.getTags() == TAGS_FMANmeasure)
			this.FMANmeasure = agregation.getSelectedTag().getID();
		}
	
	
	
	 
    //public static final int coeff1 = 1;	// Default: Vote pondéré
    //public static final int coeff2 = 2;	// Vote majoritaire
    
    
        
    //private int Coefficient = coeff1;
   
    
    //public static final Tag [] TAGS_Coefficient = {
		//new Tag(coeff1, "1"),
		//new Tag(coeff2, "2"),
		//};

       
    //public SelectedTag getCoeff_Mesure() {
		//return new SelectedTag(Coefficient, TAGS_Coefficient);
		//}

	//public void setCoeff_Mesure(SelectedTag agregation) {
		//if (agregation.getTags() == TAGS_Coefficient)
			//this.Coefficient = agregation.getSelectedTag().getID();
		//}
	/**
     * Le choix de la technique du vote 
     * en cas où nous avons retenu tout les valeurs nominales
     * de l'attribut qui maximise le gain Informationel.
     */
    
    public static final int Vote_Pond = 1;	// Default: Vote pondéré
    public static final int Vote_Maj = 2;	// Vote majoritaire
    
    
        
    private int VoteMethods = Vote_Maj;
   
    
    public static final Tag [] TAGS_VoteMethods = {
		new Tag(Vote_Pond, "Ponderat Vote"),
		new Tag(Vote_Maj, "Majority Vote"),
		};

       
    public SelectedTag getVote_Methods() {
		return new SelectedTag(VoteMethods, TAGS_VoteMethods);
		}

	public void setVote_Methods(SelectedTag agregation) {
		if (agregation.getTags() == TAGS_VoteMethods)
			this.VoteMethods = agregation.getSelectedTag().getID();
		}
	
		/**
	 * Un filtre permettant de transformer les données numériques en données nominales
	 */

	protected static Filter m_Filter = new weka.filters.unsupervised.attribute.Discretize();
	
	public void setFilter(Filter filter) {
		m_Filter = filter;		
		}
    
    public Filter getFilter() {
    	return m_Filter;     
    	}
    
    protected String getFilterSpec() {
        
        Filter c = getFilter();
        if (c instanceof OptionHandler) {
            return c.getClass().getName() + " "
                    + Utils.joinOptions(((OptionHandler)c).getOptions());
        }
        return c.getClass().getName();
    }
    
    /** The instance structure of the filtered instances */
    protected Instances m_FilteredInstances;
	
    /**
	 * Un filtre permettant de tester les mesures
	 */

	 
    
    
	 /**
	   * Returns an instance of a TechnicalInformation object, containing detailed
	   * information about the technical background of this class, e.g., paper
	   * reference or book this class is based on.
	   * 
	   * @return the technical information about this class
	   */
	@Override 
	public TechnicalInformation getTechnicalInformation() {
	    TechnicalInformation 	result;
	    
	    result = new TechnicalInformation(Type.INPROCEEDINGS);
	    result.setValue(Field.AUTHOR, "MEDDOURI Nida");
	    result.setValue(Field.YEAR, "2011");
	    result.setValue(Field.AUTHOR, "Nida MEDDOURI");
		result.setValue(Field.PUBLISHER,"Nida MEDDOURI");
		result.setValue(Field.ADDRESS, "Tunis, Tunisia");
		result.setValue(Field.ORGANIZATION, "Research Unit on Programming, Algorithmics and Heuristics - URPAH");
		result.setValue(Field.SCHOOL, "Faculty of Science of Tunis – FST ");
		result.setValue(Field.COPYRIGHT,"(c) MEDDOURI. All rights reserved");
		result.setValue(Field.NOTE,"Meddouri Software provides full range of Data Mining technologies\n"+ 
				"For more information on licensing Data Mining technologies\n"+ 	
				"from MEDDOURI Software, please contact nida.meddouri@gmail.com");  
		
		TechnicalInformation 	additional;
	    
	    result = new TechnicalInformation(Type.INCOLLECTION);
	    result.setValue(Field.AUTHOR, "J. Platt");
	    result.setValue(Field.YEAR, "1998");
	    result.setValue(Field.TITLE, "Fast Training of Support Vector Machines using Sequential Minimal Optimization");
	    result.setValue(Field.BOOKTITLE, "Advances in Kernel Methods - Support Vector Learning");
	    result.setValue(Field.EDITOR, "B. Schoelkopf and C. Burges and A. Smola");
	    result.setValue(Field.PUBLISHER, "MIT Press");
	    result.setValue(Field.URL, "http://research.microsoft.com/~jplatt/smo.html");
	    result.setValue(Field.PDF, "http://research.microsoft.com/~jplatt/smo-book.pdf");
	    result.setValue(Field.PS, "http://research.microsoft.com/~jplatt/smo-book.ps.gz");
	    
	    additional = result.add(Type.ARTICLE);
	    additional.setValue(Field.AUTHOR, "S.S. Keerthi and S.K. Shevade and C. Bhattacharyya and K.R.K. Murthy");
	    additional.setValue(Field.YEAR, "2001");
	    additional.setValue(Field.TITLE, "Improvements to Platt's SMO Algorithm for SVM Classifier Design");
	    additional.setValue(Field.JOURNAL, "Neural Computation");
	    additional.setValue(Field.VOLUME, "13");
	    additional.setValue(Field.NUMBER, "3");
	    additional.setValue(Field.PAGES, "637-649");
	    additional.setValue(Field.PS, "http://guppy.mpe.nus.edu.sg/~mpessk/svm/smo_mod_nc.ps.gz");
	    
	    additional = result.add(Type.INPROCEEDINGS);
	    additional.setValue(Field.AUTHOR, "Trevor Hastie and Robert Tibshirani");
	    additional.setValue(Field.YEAR, "1998");
	    additional.setValue(Field.TITLE, "Classification by Pairwise Coupling");
	    additional.setValue(Field.BOOKTITLE, "Advances in Neural Information Processing Systems");
	    additional.setValue(Field.VOLUME, "10");
	    additional.setValue(Field.PUBLISHER, "MIT Press");
	    additional.setValue(Field.EDITOR, "Michael I. Jordan and Michael J. Kearns and Sara A. Solla");
	    additional.setValue(Field.PS, "http://www-stat.stanford.edu/~hastie/Papers/2class.ps");
	    
	    return result;
	  }
	
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
    public String globalInfo() {    
	  return "Classifier Nominal Concepts to discover classification Rules, Nida MEDDOURI & Al (2010)." 
			  
     + "algorithm for training a support vector classifier.\n\n"
     + "This implementation globally replaces all missing values and "
     + "transforms nominal attributes into binary ones. It also "
     + "normalizes all attributes by default. (In that case the coefficients "
     + "in the output are based on the normalized data, not the "
     + "original data --- this is important for interpreting the classifier.)\n\n"
     + "Multi-class problems are solved using pairwise classification "
     + "(1-vs-1 and if logistic models are built pairwise coupling "
     + "according to Hastie and Tibshirani, 1998).\n\n"
     + "To obtain proper probability estimates, use the option that fits "
     + "logistic regression models to the outputs of the support vector "
     + "machine. In the multi-class case the predicted probabilities "
     + "are coupled using Hastie and Tibshirani's pairwise coupling "
     + "method.\n\n"
     + "Note: for improved speed normalization should be turned off when "
     + "operating on SparseInstances.\n\n"
     
			  + "For more information, see:\n\n"
			  + getTechnicalInformation().toString();  
	  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
	public Capabilities getCapabilities(){
	  
		Capabilities result = super.getCapabilities();
	    result.disableAll();

		result.enable(Capability.BINARY_ATTRIBUTES);		
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		
	    result.disableAllClasses();
	    result.disableAllClassDependencies();
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.BINARY_CLASS);
		//result.enable(Capability.NUMERIC_CLASS);
	    result.enable(Capability.MISSING_CLASS_VALUES);
	    
	    result.enable(Capability.MISSING_VALUES);
			 
		result.setMinimumNumberInstances(1);
		
		return result;
  }

	/**
	   * Returns an enumeration describing the available options.
	   *
	   * @return an enumeration of all the available options.
	   */
	  public Enumeration listOptions() {

		  Vector newVector = new Vector(2);
		  newVector.addElement(new Option(
				  "\tFull class name of filter to use, followed\n"
						  + "\tby filter options.\n"
						  + "\teg: \"weka.filters.unsupervised.attribute.Remove -V -R 1,2\"",
						  "F", 1, "-F <filter specification>"));

		  Enumeration enu = super.listOptions();
		  while (enu.hasMoreElements()) {
			  newVector.addElement(enu.nextElement());
		  }
	    
		  return newVector.elements();
	  }

	  /*
	   * Parses a given list of options. <p/>
	   *
	   <!-- options-start -->
	   * Valid options are: <p/>
	   * 
	   * <pre> -Q
	   *  Use resampling for boosting.</pre>
	   * 
	   * <pre> -I &lt;num&gt;
	   *  Number of iterations.
	   *  (default 10)</pre>
	   * 
	   * <pre> -D
	   *  If set, classifier is run in debug mode and
	   *  may output additional info to the console</pre>
	   * 
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
		  
		  boolean  runString;	  	  
		  
		    // Fermeture du meilleur attribut nominal	
			runString = Utils.getFlag("CONCEPT_LEARNING_FMAN", options);
			if (runString)
				NominalConceptLearning = CONCEPT_LEARNING_FMAN;
			
			switch (NominalConceptLearning) { 
			case CONCEPT_LEARNING_FMAN:    NominalConceptLearning = 1; break;
		    }
			
			//Les metriques de la selection du meilleur attribut nominal
			runString = Utils.getFlag("FMAN_GAIN_INFO_BV", options);
			if (runString)
				FMANmeasure = FMAN_GAIN_INFO_BV;
					
			runString = Utils.getFlag("FMAN_GAIN_INFO_BA", options);
			if (runString)
				FMANmeasure = FMAN_GAIN_INFO_BA;
			
			runString = Utils.getFlag("FMAN_GAIN_INFO_MV", options);
			if (runString)
				FMANmeasure = FMAN_GAIN_INFO_MV;
			
			runString = Utils.getFlag("FMAN_GAIN_RATIO", options);
			if (runString)
				FMANmeasure = FMAN_GAIN_RATIO;
			
			runString = Utils.getFlag("FMAN_ONE_R", options);
			if (runString)
				FMANmeasure = FMAN_ONE_R;
			
			runString = Utils.getFlag("FMAN_Correlation_Att_Eval", options);
			if (runString)
				FMANmeasure = FMAN_Correlation_Att_Eval;
			
			runString = Utils.getFlag("FMAN_Symmetrical", options);
			if (runString)
				FMANmeasure = FMAN_Symmetrical;
			
			runString = Utils.getFlag("FMAN_ClassifierAttributeEval", options);
			if (runString)
				FMANmeasure = FMAN_ClassifierAttributeEval;
			
			runString = Utils.getFlag("FMAN_ReliefFAttributeEval", options);
			if (runString)
				FMANmeasure = FMAN_ReliefFAttributeEval;
			
			runString = Utils.getFlag("FMAN_PrincipalComponents", options);
			if (runString)
				FMANmeasure = FMAN_PrincipalComponents;
			
			runString = Utils.getFlag("FMAN_InformationMutuelle", options);
			if (runString)
				FMANmeasure = FMAN_InformationMutuelle;
			
				
			runString = Utils.getFlag("FMAN_HRATIO", options);
			if (runString)
				FMANmeasure = FMAN_HRATIO;
			
			runString = Utils.getFlag("FMAN_MeanCORR_RATIO", options);
			if (runString)
				FMANmeasure = FMAN_MeanCORR_RATIO;
							
			switch (FMANmeasure) { 
			 case FMAN_GAIN_INFO_BV           :	         FMANmeasure = 1; break;
			 case FMAN_GAIN_INFO_BA           :	         FMANmeasure = 2; break;
			 case FMAN_GAIN_INFO_MV           :	         FMANmeasure = 3; break;
			 case FMAN_GAIN_RATIO             :          FMANmeasure = 4; break;
			 case FMAN_ONE_R                  :          FMANmeasure = 5; break;
			 case FMAN_Correlation_Att_Eval   :          FMANmeasure = 6; break;
			 case FMAN_Symmetrical            :          FMANmeasure = 7; break;
			 case FMAN_ClassifierAttributeEval:          FMANmeasure = 8; break;
			 case FMAN_ReliefFAttributeEval   :          FMANmeasure = 9; break;
			 case FMAN_PrincipalComponents    :          FMANmeasure = 10;break;
			 case FMAN_InformationMutuelle:              FMANmeasure = 11;break;
			 case FMAN_MeanCORR_RATIO:                   FMANmeasure = 12;break;
			 case FMAN_HRATIO:                           FMANmeasure = 13;break;
		     }	  
			
			// Les techniques de vote dans le cas de la fermeture des valeurs 
			// nominales de l'attribut qui m'aximise le gain informationel
			runString = Utils.getFlag("Vote_Pond", options);
			if ((FMANmeasure == FMAN_GAIN_INFO_MV) && runString)
				VoteMethods = Vote_Pond;
					
			runString = Utils.getFlag("Vote_Maj", options);
			if ((FMANmeasure == FMAN_GAIN_INFO_MV) && runString)
				VoteMethods = Vote_Maj;
					
			switch (VoteMethods) { 
			 case Vote_Pond:	VoteMethods = 1; break;
			 case Vote_Maj:		VoteMethods = 2; break;
		     }	 
			
			
			//runString = Utils.getFlag("coeff1", options);
			//if ((FMANmeasure == FMAN_GAIN_RATIO && runString))
			//	Coefficient = coeff1;
			//runString = Utils.getFlag("coeff2", options);
			//if ((FMANmeasure == FMAN_GAIN_RATIO && runString))
				//Coefficient = coeff2;
					
								
			//switch (Coefficient) { 
			 //case coeff1:	Coefficient = 1; break;
			 //case coeff2:	Coefficient = 2; break;
		     //}	
		    // Same for filter
		    String filterString = Utils.getOption('F', options);
		    if (filterString.length() > 0) {
		      String [] filterSpec = Utils.splitOptions(filterString);
		      if (filterSpec.length == 0) {
			throw new IllegalArgumentException("Invalid filter specification string");
		      }
		      String filterName = filterSpec[0];
		      filterSpec[0] = "";
		      setFilter((Filter) Utils.forName(Filter.class, filterName, filterSpec));
		    } else {
		      setFilter(new weka.filters.supervised.attribute.Discretize());
		    }

		    super.setOptions(options);			
	  }

	  /**
	   * Gets the current settings of the Classifier.
	   *
	   * @return an array of strings suitable for passing to setOptions
	   */
	  public String[] getOptions() 
	  {	  
		  ArrayList <String> result =new ArrayList <String>();  
		  
		  switch(NominalConceptLearning) 
		  {
		  case CONCEPT_LEARNING_FMAN: 	result.add("-fman"); break;
		  }

		  if(NominalConceptLearning == CONCEPT_LEARNING_FMAN)
			  switch(FMANmeasure) 
			  {
			  case FMAN_GAIN_INFO_BV:	
				  result.add("-giBestV"); break;	
				  
			  case FMAN_GAIN_RATIO:	
				  result.add("-giRatioBestV"); 
				  //switch(Coefficient) 
				  //{
				  //case coeff1:	result.add("-coeff"); break;
				  //case coeff2:	result.add("-coeff"); break; 
				  //}
				  break;
				 				  				  
			  case FMAN_ONE_R:	
				  result.add("-giOneRBestV"); break;
				  
			  case FMAN_Correlation_Att_Eval:
			      result.add("-giCorrelationBestV");break;
			  
			  case FMAN_Symmetrical:
			      result.add("-giSymmetricalBestV");break;
			      
			  case FMAN_ClassifierAttributeEval:
				  result.add("-giClassifierAttributeEval");break;
				  
			  case FMAN_ReliefFAttributeEval:		  
				  result.add("-giReliefFAttEval");break;
		      
			  case FMAN_PrincipalComponents:
				  result.add("-giPrincipalComponents");break;
				  
			  case FMAN_InformationMutuelle:
				  result.add("-giInformationMutuelle");break;
					  
			  case FMAN_HRATIO:
				  result.add("-giHRATIO");break;
				  
			  case FMAN_MeanCORR_RATIO:
				  result.add("-giMoyenneCorr&Ratio");break;  
			      
			  case FMAN_GAIN_INFO_BA:	
				  result.add("-giBestA"); break;	
				  
			  case FMAN_GAIN_INFO_MV:	
				  result.add("-giMultiV"); 
				  switch(VoteMethods) 
				  {
				  case Vote_Pond:	result.add("-pondVote"); break;
				  case Vote_Maj:	result.add("-majVote"); break; 
				  }
				  break;
			  }
		  
		  if(m_Debug)
				 result.add("-MODE_DEBUG");		// Mode Debug
		  
		  result.add(getFilterSpec());
			    
		  return (String[]) result.toArray(new String[result.size()]);
	  }
		
	  public String generationTipText() { 
		  return "If set to rules, Classifier Nominal Concept may output  in the log file all rules generated." ; 
	  }
  
  
    
  
     
  /**
   * Classifies a given instance.
   * 
   * @param inst the instance to be classified
   * @return the classification of the instance
   */
  @Override
  public double classifyInstance(Instance inst) throws Exception  {
	  // default model?
	  /*if (m_ZeroR != null) 
		  return m_ZeroR.classifyInstance(inst);
	  */
	  
    //System.err.println("FilteredClassifier:: " + m_Filter.getClass().getName() + " in: " + inst);

    if (m_Filter.numPendingOutput() > 0) {
      throw new Exception("Filter output queue not empty!");
    }
    
    if (!m_Filter.input(inst)) {
      throw new Exception("Filter didn't make the test instance immediately available!");
    }
    m_Filter.batchFinished();
    Instance newInstance = m_Filter.output();

    //System.err.println("FilteredClassifier:: " + m_Filter.getClass().getName() + " out: " + newInstance); 
    
    m_Missing.input(inst);
    m_Missing.batchFinished();
    inst = m_Missing.output();

	  double result= (double) -1.0;
	  Classify_Instance  listRules = new Classify_Instance();
	  switch(FMANmeasure)
	  {
	  case FMAN_GAIN_INFO_BA : 
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case FMAN_GAIN_RATIO : 
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case FMAN_ONE_R : 
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case FMAN_Correlation_Att_Eval:
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case FMAN_Symmetrical:
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case FMAN_ClassifierAttributeEval:
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case FMAN_ReliefFAttributeEval:
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case FMAN_PrincipalComponents:
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;   
	  case FMAN_InformationMutuelle:
		 result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  
	  case FMAN_HRATIO:
			 result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
		  
	  case FMAN_MeanCORR_RATIO:
			 result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
			  break;
			  
	  case FMAN_GAIN_INFO_BV : 
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case FMAN_GAIN_INFO_MV : 
		  switch(VoteMethods)
		  {
		  case Vote_Pond: result = (double) listRules.classify_Instance_nom_VotePond(newInstance, m_classifierNC,newInstance.numClasses()); break;
		  case Vote_Maj: result = (double) listRules.classify_Instance_nom_VoteMaj(newInstance, m_classifierNC,newInstance.numClasses()); break;
		  }
		  break;
		  
	  }
	  
	  if (result == -1.0) 		
		  return Utils.missingValue();
	  
	  return result;
  }
  
  /**
   * Classifier Nominal Concept method.
   *
   * @param data the training data to be used for generating the
   * nominal classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances instances) throws Exception {
	      
	  // can classifier handle the data?
	  getCapabilities().testWithFail(instances);
	
	  // remove instances with missing class
	  m_FilteredInstances = new Instances(instances);
	  m_FilteredInstances.deleteWithMissingClass();
	  
      if (m_FilteredInstances.numInstances() == 0)
          throw new Exception("No training instances left after removing instances with MissingClass!");
      
      /*
      m_Missing = new ReplaceMissingValues();
      m_Missing.setInputFormat(instances);
      instances = Filter.useFilter(instances, m_Missing); 
      */
      m_Missing.setInputFormat(m_FilteredInstances);
      m_FilteredInstances = Filter.useFilter(m_FilteredInstances, m_Missing);
      
      if (m_FilteredInstances.numInstances() == 0)
          throw new Exception("No training instances left after removing instances with MissingValues!");
     	  
      m_Filter.setInputFormat(m_FilteredInstances);  // filter capabilities are checked here
	  m_FilteredInstances = Filter.useFilter(m_FilteredInstances, m_Filter);

	    
	  // only class? -> build ZeroR model!! si on un seul attribut on appelle le classifieur zeroR
	  if (m_FilteredInstances.numAttributes() == 1) 
	  {
		  System.err.println("Cannot build model (only class attribute present in data!), "
	       + "using ZeroR model instead!");
		  m_ZeroR = new weka.classifiers.rules.ZeroR();
		  m_ZeroR.buildClassifier(m_FilteredInstances);
		  return;
	  } 
	  else {
	    m_ZeroR = null;
	    this.m_CNC = 1; // build CNC model
	    }
	    
	  
	  
	  switch(this.NominalConceptLearning)
	  {
	  case CONCEPT_LEARNING_FMAN: 
		  if(m_Debug){
			  calendar = Calendar.getInstance();
			  System.out.println("\n \t"+sdf.format(calendar.getTime()));  
		  }
		  //for (int i=0; i<m_FilteredInstances.numInstances();i++)
			  //System.out.println(m_FilteredInstances.instance(i).toString());
		  buildClassifierWithNominalClosure(m_FilteredInstances);  
		  break;
	  }  
  }
  
  
  protected void buildClassifierWithNominalClosure(Instances LearningData) throws Exception {

	  m_classifierNC = new ArrayList <Classification_Rule> ();	
	  m_classifierNC.clear();
	  
	  switch (this.FMANmeasure) 
	  {
	  case FMAN_GAIN_INFO_BV: // Fermeture de la valeur nominale la plus pertienente (Support) de l'attribut nominal qui maximise le Gain Informationel 
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 1);
		  break; 
		
	  case FMAN_GAIN_INFO_BA: // Fermeture du Meilleur Attribut Nominal selon les classes
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 2);
		  break;
		    
	  case FMAN_GAIN_INFO_MV: // Fermetures des valeurs nominales de l'attribut nominal qui maximise le Gain Informatioonel
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 3);
		  break;
	  
	  case FMAN_GAIN_RATIO: // Fermetures des valeurs nominales de l'attribut nominal qui maximise LE GAIN RATIO
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 4);
		  break;
		  
	  case FMAN_ONE_R: // Fermetures des valeurs nominales de l'attribut nominal qui maximise LE ONE R
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 5);
		  break;
		  
	  case FMAN_Correlation_Att_Eval: // Fermetures des valeurs nominales de l'attribut nominal qui maximise la correlation
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 6);
		  break;
		  
	  case FMAN_Symmetrical: // Fermetures des valeurs nominales de l'attribut nominal qui maximise LE symmetricalUncertattribute
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 7);
		  break;
		  
	  case FMAN_ClassifierAttributeEval: // Fermetures des valeurs nominales de l'attribut nominal qui maximise LE ClassifierAttributeEval
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 8);
		  break;
		  
	  case FMAN_ReliefFAttributeEval: // Fermetures des valeurs nominales de l'attribut nominal qui maximise LE ReliefFAttributeEval
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 9);
		  break;
		  
	  case FMAN_PrincipalComponents: // Fermetures des valeurs nominales de l'attribut nominal qui maximise LE PrincipalComponents
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 10);
		  break;  
		  
	  case FMAN_InformationMutuelle: // Fermetures des valeurs nominales de l'attribut nominal qui maximise LE ChiSquaredAttributeEval
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 11);
		  break;
		  
	  case FMAN_MeanCORR_RATIO: // Fermetures des valeurs nominales de l'attribut nominal qui maximise LE ChiSquaredAttributeEval
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 12);
		  break;
		  
	  
	  case FMAN_HRATIO: // Fermetures des valeurs nominales de l'attribut nominal qui maximise La mesure B1  (b+c)/a
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 13);
		  break;
	  }
	  
	  if(m_Debug)	{
		  System.out.println("\n\n\t=== Vector CLASSIFIER NOMINAL CONCEPT ===");
		  for(int i=0; i<m_classifierNC.size();i++)
			  System.out.println("CNC["+i+"]: "+m_classifierNC.get(i).affich_nom_rule(true));
		}
}  
  
 
  public ArrayList <String > extraireInstances (_Rules rgl) throws Exception {
	    ArrayList <String > temp = new ArrayList <String>() ;
	    temp= rgl.gettab_attr_regle();	// Extraire la liste des valeurs des attributs
	    String indClassMaj = new String ();
	    indClassMaj = String.valueOf(rgl.ClasseMajoritaireRegle);	//Extraire l'indice de la classe majoritaire
	    temp.add(indClassMaj);
	    return temp;
  }
  
  public ArrayList<Classification_Rule> ExtraireRegleFermNom(Instances inst, int critere) throws Exception {
	  
		ArrayList <Classification_Rule> classifierNC= new ArrayList<Classification_Rule>();
				
		if(m_Debug)
		{
			System.out.println("\nAffichage du context non binaire");
			System.out.println("\tListe des attributs:");
			System.out.print("\t");
			for(int i=0; i<inst.numAttributes();i++)
				System.out.print("("+(i+1)+")"+inst.attribute(i).name()+"  ");
			System.out.println("\n\tContext:");
			for (int i=0 ; i<inst.numInstances(); i++)
				//System.out.println("\t"+(i+1)+" : "+inst.instance(i).toString());
				System.out.println("\t"+inst.instance(i).toString());
		}
		
		m_InfoGainAttributeEval.buildEvaluator(inst);
		
		// Compute attribute with maximum information gain (FROM ID3).
	    double[] infoGains = new double[inst.numAttributes()];
	    Enumeration attEnum = inst.enumerateAttributes();
	    while (attEnum.hasMoreElements()) {
	      Attribute att = (Attribute) attEnum.nextElement();
	      //infoGains[att.index()] = computeInfoGain(inst, att);
	      infoGains[att.index()] = m_InfoGainAttributeEval.evaluateAttribute(att.index());
		    }   
	    
	    if(m_Debug){
	    	System.out.println("\nCalcul des gains informationels de chaque attribut de ce context");
		    for(int i=0; i<inst.numAttributes();i++)
		    	System.out.println("\tInfoGains de l'attribut "+inst.attribute(i).name()+": "+infoGains[i]);
	    }
	    
	    Attribute m_Attribute;
	    m_Attribute = inst.attribute(Utils.maxIndex(infoGains));
	    if(m_Debug){	
	    	System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_Attribute.name());
	    	System.out.println("\tAttribut d'indice "+m_Attribute.index());
	    	System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_Attribute.index()));
	    	for(int i=0; i<inst.numDistinctValues(m_Attribute.index()); i++)
	    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_Attribute.index()).value(i));
	    }
	    
	    /* Nida Saint Etienne 20201008

     // COMPUTE GAIN RATIO
	m_GainRatioAttributeEval.buildEvaluator(inst);
		
		// Compute attribute with maximum gain ratio.
	    double[] GainRatio = new double[inst.numAttributes()];
	    Enumeration attEnumRATIO = inst.enumerateAttributes();
	    while (attEnumRATIO.hasMoreElements()) {
	      Attribute att = (Attribute) attEnumRATIO.nextElement();
	      GainRatio[att.index()] = m_GainRatioAttributeEval.evaluateAttribute(att.index());
	    		 
		    }   
	    
	    if(m_Debug){
	    	System.out.println("\nCalcul des gains ratio de chaque attribut de ce context");
		    for(int i=0; i<inst.numAttributes();i++)
		    	System.out.println("\tGain ratio de l'attribut "+inst.attribute(i).name()+": "+GainRatio[i]);
	    }
	    
	    Attribute m_AttributeRatio;
	    m_AttributeRatio = inst.attribute(Utils.maxIndex(GainRatio));
	    if(m_Debug){	
	    	System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_AttributeRatio.name());
	    	System.out.println("\tAttribut d'indice "+m_AttributeRatio.index());
	    	System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_AttributeRatio.index()));
	    	for(int i=0; i<inst.numDistinctValues(m_AttributeRatio.index()); i++)
	    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_AttributeRatio.index()).value(i));
	    }
	    
	    // COMPUTE moyenne de gain ratio et correlation
		m_GainRatioAttributeEval.buildEvaluator(inst);
			
			// Compute attribute with maximum gain ratio.
		    double[] GainRatio1 = new double[inst.numAttributes()];
		    Enumeration attEnumRATIO1 = inst.enumerateAttributes();
		    while (attEnumRATIO1.hasMoreElements()) {
		      Attribute att = (Attribute) attEnumRATIO1.nextElement();
		      GainRatio1[att.index()] = m_GainRatioAttributeEval.evaluateAttribute(att.index());
		    		 
			    }   
		    
		    if(m_Debug){
		    	System.out.println("\nCalcul des gains ratio de chaque attribut de ce contexte");
			    for(int i=0; i<inst.numAttributes();i++)
			    	System.out.println("\tGain ratio de l'attribut "+inst.attribute(i).name()+": "+GainRatio1[i]);
		    }
		    
		    
        m_CorrelationAttributeEval.buildEvaluator(inst);
			
			// Compute attribute with maximum correlation
		    double[] CorrelationAtt = new double[inst.numAttributes()];
		    Enumeration attEnumCorrelation = inst.enumerateAttributes();
		    while (attEnumCorrelation.hasMoreElements()) {
		      Attribute att = (Attribute) attEnumCorrelation.nextElement();
		      CorrelationAtt[att.index()] = m_CorrelationAttributeEval.evaluateAttribute(att.index());
		    		 
			    }   
		    
		    if(m_Debug){
		    	System.out.println("\nCalcul des correlation de chaque attribut de ce contexte");
			    for(int i=0; i<inst.numAttributes();i++)
			    	System.out.println("\tOne R de l'attribut "+inst.attribute(i).name()+": "+CorrelationAtt[i]);
		    }
		    double[] Moyenne = new double[inst.numAttributes()];
		    for(int i=0; i<inst.numAttributes();i++)
		    	Moyenne[i]= (GainRatio[i]+ CorrelationAtt[i])/2;
		    	

		   	    		
		   Attribute m_AttributeCorreRatio;
		           m_AttributeCorreRatio = inst.attribute(Utils.maxIndex(Moyenne));
		           if(m_Debug){	
				    	System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_AttributeCorreRatio.name());
				    	System.out.println("\tAttribut d'indice "+m_AttributeCorreRatio.index());
				    	System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_AttributeCorreRatio.index()));
				    	for(int i=0; i<inst.numDistinctValues(m_AttributeCorreRatio.index()); i++)
				    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_AttributeCorreRatio.index()).value(i));
				    }
			
	 // COMPUTE ONE R
		m_OneRAttributeEval.buildEvaluator(inst);
			
			// Compute attribute with maximum ONE_R
		    double[] ONE_R = new double[inst.numAttributes()];
		    Enumeration attEnumONER = inst.enumerateAttributes();
		    while (attEnumONER.hasMoreElements()) {
		      Attribute att = (Attribute) attEnumONER.nextElement();
		      ONE_R[att.index()] = m_OneRAttributeEval.evaluateAttribute(att.index());
		    		 
			    }   
		    
		    if(m_Debug){
		    	System.out.println("\nCalcul des One R de chaque attribut de ce context");
			    for(int i=0; i<inst.numAttributes();i++)
			    	System.out.println("\tOne R de l'attribut "+inst.attribute(i).name()+": "+ONE_R[i]);
		    }
		    
		    Attribute m_AttributeONE_R;
		    m_AttributeONE_R = inst.attribute(Utils.maxIndex(ONE_R));
		    if(m_Debug){	
		    	System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_AttributeONE_R.name());
		    	System.out.println("\tAttribut d'indice "+m_AttributeONE_R.index());
		    	System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_AttributeONE_R.index()));
		    	for(int i=0; i<inst.numDistinctValues(m_AttributeONE_R.index()); i++)
		    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_AttributeONE_R.index()).value(i));
		    }
		    // COMPUTE Correlation
			m_CorrelationAttributeEval.buildEvaluator(inst);
				
				// Compute attribute with maximum correlation
			    double[] CorrelationAtt1 = new double[inst.numAttributes()];
			    Enumeration attEnumCorrelation1 = inst.enumerateAttributes();
			    while (attEnumCorrelation1.hasMoreElements()) {
			      Attribute att = (Attribute) attEnumCorrelation1.nextElement();
			      CorrelationAtt1[att.index()] = m_CorrelationAttributeEval.evaluateAttribute(att.index());
			    		 
				    }   
			    
			    if(m_Debug){
			    	System.out.println("\nCalcul des One R de chaque attribut de ce context");
				    for(int i=0; i<inst.numAttributes();i++)
				    	System.out.println("\tOne R de l'attribut "+inst.attribute(i).name()+": "+CorrelationAtt1[i]);
			    }
			    
			    Attribute m_AttributeCorrelation1;
			    m_AttributeCorrelation1 = inst.attribute(Utils.maxIndex(CorrelationAtt1));
			    if(m_Debug){	
			    	System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_AttributeCorrelation1.name());
			    	System.out.println("\tAttribut d'indice "+m_AttributeCorrelation1.index());
			    	System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_AttributeCorrelation1.index()));
			    	for(int i=0; i<inst.numDistinctValues(m_AttributeCorrelation1.index()); i++)
			    		System.out.println("hello world\t"+inst.attribute(m_AttributeCorrelation1.index()).value(i));
			    	//	System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_AttributeCorrelation.index()).value(i));
			    }
			    
///////////
			    
			 // COMPUTE Symmetrical
				m_SymmetricalUncertAttributeEval.buildEvaluator(inst);
					
					// Compute attribute with maximum symmetrical
				    double[] Symmetrical = new double[inst.numAttributes()];
				    Enumeration attEnumSymmetrical = inst.enumerateAttributes();
				    while (attEnumSymmetrical.hasMoreElements()) {
				      Attribute att = (Attribute) attEnumSymmetrical.nextElement();
				       Symmetrical[att.index()] = m_SymmetricalUncertAttributeEval.evaluateAttribute(att.index());
				    		 
					    }   
				    
				    if(m_Debug){
				    	System.out.println("\nCalcul des One R de chaque attribut de ce context");
					    for(int i=0; i<inst.numAttributes();i++)
					    	System.out.println("\tOne R de l'attribut "+inst.attribute(i).name()+": "+Symmetrical[i]);
				    }
				    
				    Attribute m_AttributeSymmetrical;
				    m_AttributeSymmetrical = inst.attribute(Utils.maxIndex(Symmetrical));
				    if(m_Debug){	
				    	System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_AttributeSymmetrical.name());
				    	System.out.println("\tAttribut d'indice "+m_AttributeSymmetrical.index());
				    	System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_AttributeSymmetrical.index()));
				    	for(int i=0; i<inst.numDistinctValues(m_AttributeSymmetrical.index()); i++)
				    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_AttributeSymmetrical.index()).value(i));
				    }
				    
				    
				    
				    // COMPUTE ClassifierAttributeEval
					m_ClassifierAttributeEval.buildEvaluator(inst);
						
						// Compute attribute with maximum ClassifierAttributeEval
					    double[] Classifier_Attribute_Eval = new double[inst.numAttributes()];
					    Enumeration attEnumClassifierAttributeEval = inst.enumerateAttributes();
					    while (attEnumClassifierAttributeEval.hasMoreElements()) {
					      Attribute att = (Attribute) attEnumClassifierAttributeEval.nextElement();
					      Classifier_Attribute_Eval[att.index()] = m_ClassifierAttributeEval.evaluateAttribute(att.index());
					    		 
						    }   
					    
					    if(m_Debug){
					    	System.out.println("\nCalcul des One R de chaque attribut de ce context");
						    for(int i=0; i<inst.numAttributes();i++)
						    	System.out.println("\tOne R de l'attribut "+inst.attribute(i).name()+": "+Classifier_Attribute_Eval[i]);
					    }
					    
					    Attribute m_AttributeClassifierAttributeEval;
					    m_AttributeClassifierAttributeEval = inst.attribute(Utils.maxIndex(Classifier_Attribute_Eval));
					    if(m_Debug){	
					    	System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_AttributeClassifierAttributeEval.name());
					    	System.out.println("\tAttribut d'indice "+m_AttributeClassifierAttributeEval.index());
					    	System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_AttributeClassifierAttributeEval.index()));
					    	for(int i=0; i<inst.numDistinctValues(m_AttributeClassifierAttributeEval.index()); i++)
					    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_AttributeClassifierAttributeEval.index()).value(i));
					    }
					    
					    // COMPUTE ReliefFAttributeEval
						m_ReliefFAttributeEval.buildEvaluator(inst);
							
							// Compute attribute with maximum symmetrical
						    double[] ReliefF_Attribute_Eval = new double[inst.numAttributes()];
						    Enumeration attEnumReliefFAttributeEval = inst.enumerateAttributes();
						    while (attEnumReliefFAttributeEval.hasMoreElements()) {
						      Attribute att = (Attribute) attEnumReliefFAttributeEval.nextElement();
						      ReliefF_Attribute_Eval[att.index()] = m_ReliefFAttributeEval.evaluateAttribute(att.index());
						    		 
							    }   
						    
						    if(m_Debug){
						    	System.out.println("\nCalcul des One R de chaque attribut de ce context");
							    for(int i=0; i<inst.numAttributes();i++)
							    	System.out.println("\tOne R de l'attribut "+inst.attribute(i).name()+": "+ReliefF_Attribute_Eval[i]);
						    }
						    
						    Attribute m_AttributeReliefFAttributeEval;
						    m_AttributeReliefFAttributeEval = inst.attribute(Utils.maxIndex(ReliefF_Attribute_Eval));
						    if(m_Debug){	
						    	System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_AttributeReliefFAttributeEval.name());
						    	System.out.println("\tAttribut d'indice "+m_AttributeReliefFAttributeEval.index());
						    	System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_AttributeReliefFAttributeEval.index()));
						    	for(int i=0; i<inst.numDistinctValues(m_AttributeReliefFAttributeEval.index()); i++)
						    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_AttributeReliefFAttributeEval.index()).value(i));
						    }
						 // COMPUTE PrincipalComponents
							m_PrincipalComponents.buildEvaluator(inst);
								
								// Compute attribute with maximum PrincipalComponents
							    double[] PrincipalComponents_Eval = new double[inst.numAttributes()];
							    Enumeration attEnumPrincipalComponents = inst.enumerateAttributes();
							    while (attEnumPrincipalComponents.hasMoreElements()) {
							      Attribute att = (Attribute) attEnumPrincipalComponents.nextElement();
							      PrincipalComponents_Eval[att.index()] = m_PrincipalComponents.evaluateAttribute(att.index());
							    		 
								    }   
							    
							    if(m_Debug){
							    	System.out.println("\nCalcul des One R de chaque attribut de ce context");
								    for(int i=0; i<inst.numAttributes();i++)
								    	System.out.println("\tPrincipalComponents de l'attribut "+inst.attribute(i).name()+": "+PrincipalComponents_Eval[i]);
							    }
							    
							    Attribute m_AttributePrincipalComponents;
							    m_AttributePrincipalComponents= inst.attribute(Utils.maxIndex(PrincipalComponents_Eval));
							    if(m_Debug){	
							    	System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_PrincipalComponents.name());
							    	System.out.println("\tAttribut d'indice "+m_AttributePrincipalComponents.index());
							    	System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_AttributePrincipalComponents.index()));
							    	for(int i=0; i<inst.numDistinctValues(m_AttributePrincipalComponents.index()); i++)
							    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_AttributePrincipalComponents.index()).value(i));
							    }
							    //COMPUTE Information mutual
			  				m_InformationMutuelle.buildEvaluator(inst);
									
									// Compute attribute with maximum Information mutuelle
							    double[] InformationMutuelle_Eval = new double[inst.numAttributes()];
							    Enumeration attEnumInformationMutuelle = inst.enumerateAttributes();
							    while (attEnumInformationMutuelle.hasMoreElements()) {
						      Attribute att = (Attribute) attEnumInformationMutuelle.nextElement();
						      InformationMutuelle_Eval[att.index()] = m_InformationMutuelle.evaluateAttribute(att.index());
								    		 
									   }   
								    
								    if(m_Debug){
								   	System.out.println("\nCalcul des info mutuelles de chaque attribut de ce context");
									   for(int i=0; i<inst.numAttributes();i++)
									  	System.out.println("\tinfo mutuelle de l'attribut "+inst.attribute(i).name()+": "+InformationMutuelle_Eval[i]);
								    }
								    
								    Attribute m_AttributeInformationMutuelle;
								    m_AttributeInformationMutuelle= inst.attribute(Utils.maxIndex(InformationMutuelle_Eval));
								    if(m_Debug){	
								          System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_InformationMutuelle.name());
								    	  System.out.println("\tAttribut d'indice "+m_AttributeInformationMutuelle.index());
								    	  System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_AttributeInformationMutuelle.index()));
								    	  for(int i=0; i<inst.numDistinctValues(m_AttributeInformationMutuelle.index()); i++)
 								    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_AttributeInformationMutuelle.index()).value(i));
								      }
								  
															  //COMPUTE H-RATIO
												  				m_HRatio.buildEvaluator(inst);
																		
																		// Compute attribute with maximum HRATIO
																    double[] HRatio_Eval = new double[inst.numAttributes()];
																    Enumeration attEnumHRatio = inst.enumerateAttributes();
																    while (attEnumHRatio.hasMoreElements()) {
															      Attribute att = (Attribute) attEnumHRatio.nextElement();
															      HRatio_Eval[att.index()] = m_HRatio .evaluateAttribute(att.index());
																	    		 
																		   }   
																	    
																	    if(m_Debug){
																	   	System.out.println("\nCalcul de la 1 ere mesure de chaque attribut de ce context");
																		   for(int i=0; i<inst.numAttributes();i++)
																		  	System.out.println("\tMesure1 de l'attribut "+inst.attribute(i).name()+": "+HRatio_Eval[i]);
																	    }
																	    
																	    Attribute m_AttributeHRatio;
																	    m_AttributeHRatio= inst.attribute(Utils.maxIndex(HRatio_Eval));
																	    if(m_Debug){	
																	          System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_HRatio.name());
																	    	  System.out.println("\tAttribut d'indice "+m_AttributeHRatio.index());
																	    	  System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_AttributeHRatio.index()));
																	    	  for(int i=0; i<inst.numDistinctValues(m_AttributeHRatio.index()); i++)
																	    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_AttributeHRatio.index()).value(i));
																	      }
																	    
		*/
		
		/* Génération d'un classifieur de type CNC à partir de la fermeture 
	     * de la valeur nominale la plus pertinente (qui maximise le Support) 
	     * de l'attribut nominal qui maximise le Gain Informationel
	     */
	    if(critere == 1) 
	    {
	    	if(m_Debug)
	    		System.out.println("\nGénération d'un CNC à partir de la fermeture du valeur la plus pertinente de l'attribut retenu");
			
	    	int supportDistVal=0;
			int indexBestDistVal = 0;
			int suppBestDistVal = 0;

			//Parcourir les differentes valeurs du 'm_Attribute'  
			for(int i=0; i<inst.numDistinctValues(m_Attribute.index()); i++)
			{				
				//Calcul du support de cette DistinctValue
				ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
				instDistVal.clear();
				
				supportDistVal=0;
				for(int j=0; j<inst.numInstances(); j++)
				{
					//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(i));
					if( inst.instance(j).stringValue(m_Attribute.index()) == inst.attribute(m_Attribute.index()).value(i))
					{
						supportDistVal++;
						instDistVal.add(j);
						//System.out.println("     OK");					
					}
					//else
						//System.out.println("     NO");
				}
				//System.out.println("Support de cette DistinctValue ("+inst.attribute(m_Attribute.index()).value(i)+"): "+supportDistVal);
				if(suppBestDistVal <= supportDistVal)
				{
					suppBestDistVal=supportDistVal;
					indexBestDistVal = i;
				}
			}	
			
			if(m_Debug)
				System.out.println("Meilleur DistinctValue: ( "+m_Attribute.value(indexBestDistVal)+" ) avec un support qui vaut: "+suppBestDistVal);
			
			//Extraires les exemples associés a cet attribut
			ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
			ArrayList <String> FERM_att = new ArrayList <String> ();	
	
			//Liste des instances verifiant la fermeture
			for(int i=0; i<inst.numInstances(); i++)
			{
				if( inst.instance(i).stringValue(m_Attribute.index()) == m_Attribute.value(indexBestDistVal))
					FERM_exe.add(i);
			}
			int nbrInstFer = FERM_exe.size();
			
			if(m_Debug)
			{
				System.out.print("Fermeture des instances: \n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
				for(int i=0; i<nbrInstFer; i++)
					System.out.print(FERM_exe.get(i)+" - ");
				System.out.println();
				
				for(int i=0; i<nbrInstFer; i++)
					System.out.println("\t\t"+FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
			}
			
			//Liste des attributs associés à la fermeture 
			String nl= "-null-";
			for(int i=0; i<(int) inst.numAttributes()-1;i++)
			{
				int cmpt=0;
				String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
				//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
				for (int j=0; j<nbrInstFer;)
				{
					if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
					{
						cmpt++;
						if(cmpt==nbrInstFer)
						{
							FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
							//System.out.println(" ---> ok ");
						}
						j++;
					}
					else
					{
						j=nbrInstFer;
						FERM_att.add(nl); 
						//System.out.println(" ---> null ");
					}
				}
			}
			
			if(m_Debug)
			{
				System.out.print("Liste des attributs nominative:         ");
				for(int i=0; i<inst.numAttributes()-1;i++)
					System.out.print(inst.attribute(i).name()+" , ");
				System.out.print("\nListe des valeurs d'attribut retenues:  ");
				for(int i=0; i<FERM_att.size(); i++)
					System.out.print(FERM_att.get(i)+" , ");
			}	
			
			///////////////Extraire la classe majoritaire associée////////////////////		
			int [] nbClasse = new int [inst.numClasses()];			
			for(int k=0 ; k<inst.numClasses() ; k++)
				nbClasse[k]=0;
				
			//Parcourir les exemples associée à ce concept
			for(int j=0;j<nbrInstFer;j++)
				nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
				
			//Detertminer l'indice de la classe associée
			int indiceMax=0;
			for(int i=0;i<inst.numClasses();i++)
				if ( nbClasse[i] > nbClasse[indiceMax] )
					indiceMax=i;			
			if(m_Debug)
				System.out.println ("\nLa Classe Majoritaire est: "+inst.attribute(inst.classIndex()).value(indiceMax));
			
			// On retourne le concept Pertinent comme un vecteur de String 
			ArrayList <String> CP= new ArrayList <String>();
			for (int i=0 ; i<(int) inst.numAttributes()-1 ; i++)
					CP.add(FERM_att.get(i));
			
			Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax),infoGains[Utils.maxIndex(infoGains)]);
			classifierNC.add(r);
			
	    }
	    
	    /* Génération d'un classifieur de type CNC à partir de la fermeture 
	     * de la valeur nominale la plus pertienente (qui maximise le Support) 
	     * de l'attribut nominal qui maximise le Gain Informationel
	     */
	    //Génération d'un classifieur faible à partir de la fermeture du meilleur attribut retenu 'm_Attribute'
	    if(critere == 2)	// FMAN_GAIN_INFO_BA: Fermeture du Meilleur Attribut Nominal selon les classes
	    {
	    	int supportDistVal=0;
			int indexBestDistVal = 0;
			int suppBestDistVal = 0;

			for(int cl=0; cl<inst.numClasses(); cl++)
			{
				supportDistVal=0;
				indexBestDistVal = 0;
				suppBestDistVal = 0;
				
					//Extraction des indices d'instances etiquitées par la classe d'indice <cl>
					ArrayList <Integer> IndTrainingbyClass = new ArrayList <Integer>();
					IndTrainingbyClass.clear();
					for(int i=0; i<inst.numInstances();i++)
						if(inst.instance(i).classValue()==cl)
							IndTrainingbyClass.add(i);
					
					//Extraction de l'échantillon d'instances etiquitées par la classe d'indice <cl>
					Instances TrainingbyClass = new Instances( inst, 0,IndTrainingbyClass.size()); 
					TrainingbyClass.delete();
				    for (int h=0; h< IndTrainingbyClass.size(); h++)
				    	TrainingbyClass.add(inst.instance(IndTrainingbyClass.get(h)));
				    if(TrainingbyClass.numInstances()==0)
				       	System.out.println("\nCAS PARTICULIER: TrainingbyClass.numInstances()==0");
				    
				    if(TrainingbyClass.numInstances()!=0)
				    {
						for(int i=0; i<TrainingbyClass.numDistinctValues(m_Attribute.index()); i++)
						{
							//Calcule du support de cette DistinctValue
							ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
							instDistVal.clear();
							
							supportDistVal=0;
							for(int j=0; j<TrainingbyClass.numInstances(); j++)
							{
								//System.out.print((j+1)+"iéme instance: "+TrainingbyClass.instance(j).stringValue(m_Attribute.index()) +" - "+ TrainingbyClass.attribute(m_Attribute.index()).value(i));
								if( TrainingbyClass.instance(j).stringValue(m_Attribute.index()) == TrainingbyClass.attribute(m_Attribute.index()).value(i))
								{
									supportDistVal++;
									instDistVal.add(j);
									//System.out.println("     OK");					
								}
								//else
									//System.out.println("     NO");
							}
							//System.out.println("Support de cette DistinctValue ("+TrainingbyClass.attribute(m_Attribute.index()).value(i)+"): "+supportDistVal);
							if(suppBestDistVal <= supportDistVal)
							{
								suppBestDistVal=supportDistVal;
								indexBestDistVal = i;
							}
						}	
						
						if(m_Debug)
							System.out.println("Indice du meilleur DistinctValue("+m_Attribute.value(indexBestDistVal)+"): "+indexBestDistVal+" avec un support qui vaut: "+suppBestDistVal);
						
						//Extraires les exemples associés a cet attribut
						ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
						ArrayList <String> FERM_att = new ArrayList <String> ();	
				
						//Liste des instances verifiant la fermeture
						for(int i=0; i<TrainingbyClass.numInstances(); i++)
						{
							if( TrainingbyClass.instance(i).stringValue(m_Attribute.index()) == m_Attribute.value(indexBestDistVal))
								FERM_exe.add(i);
						}
						int nbrInstFer = FERM_exe.size();

						if(m_Debug)
						{
							System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
							for(int i=0; i<nbrInstFer; i++)
								System.out.print (" - "+FERM_exe.get(i));
							System.out.println();
							for(int i=0; i<nbrInstFer; i++)
								System.out.println(FERM_exe.get(i)+" : "+TrainingbyClass.instance(FERM_exe.get(i)).toString());
						}
						
						//Liste des attributs associés à la fermeture ??????????????????????
						//System.out.println("Extraction des attributs associés à cette fermeture");
						String nl= "-null-";
						for(int i=0; i< (int) inst.numAttributes()-1;i++)
						{
							int cmpt=0;
							String FirstDistVal = TrainingbyClass.instance(FERM_exe.get(0)).stringValue(i);
							//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
							for (int j=0; j<nbrInstFer;)
							{
								if(TrainingbyClass.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
								{
									cmpt++;
									if(cmpt==nbrInstFer)
									{
										FERM_att.add(TrainingbyClass.instance(FERM_exe.get(0)).stringValue(i));
										//System.out.println(" ---> ok ");
									}
									j++;
								}
								else
								{
									j=nbrInstFer;
									FERM_att.add(nl); 
									//System.out.println(" ---> null ");
								}
							}
						}
						
						if(m_Debug)
						{
						System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
							for(int i=0; i<FERM_att.size(); i++)
								System.out.print (FERM_att.get(i)+" , ");
						}
													
						//On retourne le concept Pertinent comme un vecteur 
						ArrayList <String> CP= new ArrayList <String>();
						for (int i=0; i< (int) inst.numAttributes()-1;i++)
								CP.add(FERM_att.get(i));
						
						Classification_Rule r = new Classification_Rule(CP,cl,inst.attribute(inst.classIndex()).value(cl));
						classifierNC.add(r);
				    }
			}
	    }

	    /* Génération d'un classifieur de type CNC à partir de la fermeture 
	     * des valeurs nominales de l'attribut nominal qui maximise le Gain Informationel
	     */
	    if(critere == 3)
	    {
	    	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
	    	
			for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_Attribute.index()); indDistVal++)
			{
				instDistVal.clear();
				
				for(int j=0; j<inst.numInstances(); j++)
				{
					//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
					if( inst.instance(j).stringValue(m_Attribute.index()) == inst.attribute(m_Attribute.index()).value(indDistVal))
					{
						instDistVal.add(j);
					//	System.out.println("     OK");					
					}
					//else
						//System.out.println("     NO");
				}
				
				if(instDistVal.size()!=0)
				{
					//Extraires les exemples associés a cet attribut
					ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
					ArrayList <String> FERM_att = new ArrayList <String> ();	
			
					//Liste des instances verifiant la fermeture
					for(int i=0; i<inst.numInstances(); i++)
					{
						if( inst.instance(i).stringValue(m_Attribute.index()) == m_Attribute.value(indDistVal))
							FERM_exe.add(i);
					}
					int nbrInstFer = FERM_exe.size();
					
					if(m_Debug)
					{	
						System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
						for(int i=0; i<nbrInstFer; i++)
							System.out.print (" - "+FERM_exe.get(i));
						System.out.println();
						for(int i=0; i<nbrInstFer; i++)
							System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
					}
					
					//Liste des attributs associés à la fermeture ??????????????????????
					//System.out.println("Extraction des attributs associés à cette fermeture");
					String nl= "-null-";
					for(int i=0; i< (int) inst.numAttributes()-1;i++)
					{
						int cmpt=0;
						String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
						//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
						for (int j=0; j<nbrInstFer;)
						{
							if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
							{
								cmpt++;
								if(cmpt==nbrInstFer)
								{
									FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
									//System.out.println(" ---> ok ");
								}
								j++;
							}
							else
							{
								j=nbrInstFer;
								FERM_att.add(nl); 
								//System.out.println(" ---> null ");
							}
						}
					}
					
					if(m_Debug)
					{
					System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
					for(int i=0; i<FERM_att.size(); i++)
						System.out.print (FERM_att.get(i)+" , ");
					}
					
					
					//Extraire la classe majoritaire associée//		
					int [] nbClasse = new int[inst.numClasses()];			
					for(int k=0;k<inst.numClasses();k++)
						nbClasse[k]=0;
						
					//Parcourir les exemples associée à ce concept
					//System.out.println();
					for(int j=0;j<nbrInstFer;j++)
						nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
						
						
					//Detertminer l'indice de la classe associé
					int indiceMax=0;
					for(int i=0;i<inst.numClasses();i++)
						if(nbClasse[i]>nbClasse[indiceMax])
							indiceMax=i;
					
					if(m_Debug)	{
						System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
						System.out.println ("Liste des des attribut de la fermeture");
						for (int o=0;o<FERM_att.size();o++)
							System.out.print (FERM_att.get(o)+" , ");
						System.out.println ("");
					}
							
					//On retourne le concept Pertinent comme un vecteur 
					ArrayList <String> CP= new ArrayList <String>();
					for (int i=0; i< (int) inst.numAttributes()-1;i++)
							CP.add(FERM_att.get(i));
					
					Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
					
					classifierNC.add(r);
					}
			}
			
	    }
	       
	    /* Génération d'un classifieur de type CNC à partir de la fermeture 
	     * des valeurs nominales de l'attribut nominal qui maximise le gain ratio
	     */
	    /*
	    if(critere == 4) 
	    {
	    	
	    	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
	    	
			for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_AttributeRatio.index()); indDistVal++)
			{
				instDistVal.clear();
				
				for(int j=0; j<inst.numInstances(); j++)
				{
					//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
					if( inst.instance(j).stringValue(m_AttributeRatio.index()) == inst.attribute(m_AttributeRatio.index()).value(indDistVal))
					{
						instDistVal.add(j);
					//	System.out.println("     OK");					
					}
					//else
						//System.out.println("     NO");
				}
				
				if(instDistVal.size()!=0)
				{
					//Extraires les exemples associés a cet attribut
					ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
					ArrayList <String> FERM_att = new ArrayList <String> ();	
			
					//Liste des instances verifiant la fermeture
					for(int i=0; i<inst.numInstances(); i++)
					{
						if( inst.instance(i).stringValue(m_AttributeRatio.index()) == m_AttributeRatio.value(indDistVal))
							FERM_exe.add(i);
					}
					int nbrInstFer = FERM_exe.size();
					
					if(m_Debug)
					{	
						System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
						for(int i=0; i<nbrInstFer; i++)
							System.out.print (" - "+FERM_exe.get(i));
						System.out.println();
						for(int i=0; i<nbrInstFer; i++)
							System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
					}
					
					//Liste des attributs associés à la fermeture ??????????????????????
					//System.out.println("Extraction des attributs associés à cette fermeture");
					String nl= "-null-";
					for(int i=0; i< (int) inst.numAttributes()-1;i++)
					{
						int cmpt=0;
						String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
						//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
						for (int j=0; j<nbrInstFer;)
						{
							if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
							{
								cmpt++;
								if(cmpt==nbrInstFer)
								{
									FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
									//System.out.println(" ---> ok ");
								}
								j++;
							}
							else
							{
								j=nbrInstFer;
								FERM_att.add(nl); 
								//System.out.println(" ---> null ");
							}
						}
					}
					
					if(m_Debug)
					{
					System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
					for(int i=0; i<FERM_att.size(); i++)
						System.out.print (FERM_att.get(i)+" , ");
					}
					
					
					//Extraire la classe majoritaire associée//		
					int [] nbClasse = new int[inst.numClasses()];			
					for(int k=0;k<inst.numClasses();k++)
						nbClasse[k]=0;
						
					//Parcourir les exemples associée à ce concept
					//System.out.println();
					for(int j=0;j<nbrInstFer;j++)
						nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
						
						
					//Detertminer l'indice de la classe associé
					int indiceMax=0;
					for(int i=0;i<inst.numClasses();i++)
						if(nbClasse[i]>nbClasse[indiceMax])
							indiceMax=i;
					
					if(m_Debug)	{
						System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
						System.out.println ("Liste des des attribut de la fermeture");
						for (int o=0;o<FERM_att.size();o++)
							System.out.print (FERM_att.get(o)+" , ");
						System.out.println ("");
					}
							
					//On retourne le concept Pertinent comme un vecteur 
					ArrayList <String> CP= new ArrayList <String>();
					for (int i=0; i< (int) inst.numAttributes()-1;i++)
							CP.add(FERM_att.get(i));
					
					Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
					
					classifierNC.add(r);
					}
			}
			
	    }
	    */   
	    
	    
 //}
    
  
  /* Génération d'un classifieur de type CNC à partir de la fermeture 
   * des valeurs nominales de l'attribut nominal qui maximise le ONER
   */
/*
  if(critere == 5) 
  {
  	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
  	
		for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_AttributeONE_R.index()); indDistVal++)
		{
			instDistVal.clear();
			
			for(int j=0; j<inst.numInstances(); j++)
			{
				//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
				if( inst.instance(j).stringValue(m_AttributeONE_R.index()) == inst.attribute(m_AttributeONE_R.index()).value(indDistVal))
				{
					instDistVal.add(j);
				//	System.out.println("     OK");					
				}
				//else
					//System.out.println("     NO");
			}
			
			if(instDistVal.size()!=0)
			{
				//Extraires les exemples associés a cet attribut
				ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
				ArrayList <String> FERM_att = new ArrayList <String> ();	
		
				//Liste des instances verifiant la fermeture
				for(int i=0; i<inst.numInstances(); i++)
				{
					if( inst.instance(i).stringValue(m_AttributeONE_R.index()) == m_AttributeONE_R.value(indDistVal))
						FERM_exe.add(i);
				}
				int nbrInstFer = FERM_exe.size();
				
				if(m_Debug)
				{	
					System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
					for(int i=0; i<nbrInstFer; i++)
						System.out.print (" - "+FERM_exe.get(i));
					System.out.println();
					for(int i=0; i<nbrInstFer; i++)
						System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
				}
				
				//Liste des attributs associés à la fermeture ??????????????????????
				//System.out.println("Extraction des attributs associés à cette fermeture");
				String nl= "-null-";
				for(int i=0; i< (int) inst.numAttributes()-1;i++)
				{
					int cmpt=0;
					String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
					//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
					for (int j=0; j<nbrInstFer;)
					{
						if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
						{
							cmpt++;
							if(cmpt==nbrInstFer)
							{
								FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
								//System.out.println(" ---> ok ");
							}
							j++;
						}
						else
						{
							j=nbrInstFer;
							FERM_att.add(nl); 
							//System.out.println(" ---> null ");
						}
					}
				}
				
				if(m_Debug)
				{
				System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
				for(int i=0; i<FERM_att.size(); i++)
					System.out.print (FERM_att.get(i)+" , ");
				}
				
				
				//Extraire la classe majoritaire associée//		
				int [] nbClasse = new int[inst.numClasses()];			
				for(int k=0;k<inst.numClasses();k++)
					nbClasse[k]=0;
					
				//Parcourir les exemples associée à ce concept
				//System.out.println();
				for(int j=0;j<nbrInstFer;j++)
					nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
					
					
				//Detertminer l'indice de la classe associé
				int indiceMax=0;
				for(int i=0;i<inst.numClasses();i++)
					if(nbClasse[i]>nbClasse[indiceMax])
						indiceMax=i;
				
				if(m_Debug)	{
					System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
					System.out.println ("Liste des des attribut de la fermeture");
					for (int o=0;o<FERM_att.size();o++)
						System.out.print (FERM_att.get(o)+" , ");
					System.out.println ("");
				}
						
				//On retourne le concept Pertinent comme un vecteur 
				ArrayList <String> CP= new ArrayList <String>();
				for (int i=0; i< (int) inst.numAttributes()-1;i++)
						CP.add(FERM_att.get(i));
				
				Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
				
				classifierNC.add(r);
				}
		}
		
  }
  */
	    
  /* Génération d'un classifieur de type CNC à partir de la fermeture 
   * des valeurs nominales de l'attribut nominal qui maximise la correlation
   */
/*
  if(critere == 6) 
  {
  	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
  	
		for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_AttributeCorrelation1.index()); indDistVal++)
		{
			instDistVal.clear();
			
			for(int j=0; j<inst.numInstances(); j++)
			{
				//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
				if( inst.instance(j).stringValue(m_AttributeCorrelation1.index()) == inst.attribute(m_AttributeCorrelation1.index()).value(indDistVal))
				{
					instDistVal.add(j);
				//	System.out.println("     OK");					
				}
				//else
					//System.out.println("     NO");
			}
			
			if(instDistVal.size()!=0)
			{
				//Extraires les exemples associés a cet attribut
				ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
				ArrayList <String> FERM_att = new ArrayList <String> ();	
		
				//Liste des instances verifiant la fermeture
				for(int i=0; i<inst.numInstances(); i++)
				{
					if( inst.instance(i).stringValue(m_AttributeCorrelation1.index()) == m_AttributeCorrelation1.value(indDistVal))
						FERM_exe.add(i);
				}
				int nbrInstFer = FERM_exe.size();
				
				if(m_Debug)
				{	
					System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
					for(int i=0; i<nbrInstFer; i++)
						System.out.print (" - "+FERM_exe.get(i));
					System.out.println();
					for(int i=0; i<nbrInstFer; i++)
						System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
				}
				
				//Liste des attributs associés à la fermeture ??????????????????????
				//System.out.println("Extraction des attributs associés à cette fermeture");
				String nl= "-null-";
				for(int i=0; i< (int) inst.numAttributes()-1;i++)
				{
					int cmpt=0;
					String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
					//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
					for (int j=0; j<nbrInstFer;)
					{
						if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
						{
							cmpt++;
							if(cmpt==nbrInstFer)
							{
								FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
								//System.out.println(" ---> ok ");
							}
							j++;
						}
						else
						{
							j=nbrInstFer;
							FERM_att.add(nl); 
							//System.out.println(" ---> null ");
						}
					}
				}
				
				if(m_Debug)
				{
				System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
				for(int i=0; i<FERM_att.size(); i++)
					System.out.print (FERM_att.get(i)+" , ");
				}
				
				
				//Extraire la classe majoritaire associée//		
				int [] nbClasse = new int[inst.numClasses()];			
				for(int k=0;k<inst.numClasses();k++)
					nbClasse[k]=0;
					
				//Parcourir les exemples associée à ce concept
				//System.out.println();
				for(int j=0;j<nbrInstFer;j++)
					nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
					
					
				//Detertminer l'indice de la classe associé
				int indiceMax=0;
				for(int i=0;i<inst.numClasses();i++)
					if(nbClasse[i]>nbClasse[indiceMax])
						indiceMax=i;
				
				if(m_Debug)	{
					System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
					System.out.println ("Liste des des attribut de la fermeture");
					for (int o=0;o<FERM_att.size();o++)
						System.out.print (FERM_att.get(o)+" , ");
					System.out.println ("");
				}
						
				//On retourne le concept Pertinent comme un vecteur 
				ArrayList <String> CP= new ArrayList <String>();
				for (int i=0; i< (int) inst.numAttributes()-1;i++)
						CP.add(FERM_att.get(i));
				
				Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
				
				classifierNC.add(r);
				}
		}
		
  }
  */
	    
/*
  if(critere == 7) 
  {
  	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
  	
		for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_AttributeSymmetrical.index()); indDistVal++)
		{
			instDistVal.clear();
			
			for(int j=0; j<inst.numInstances(); j++)
			{
				//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
				if( inst.instance(j).stringValue(m_AttributeSymmetrical.index()) == inst.attribute(m_AttributeSymmetrical.index()).value(indDistVal))
				{
					instDistVal.add(j);
				//	System.out.println("     OK");					
				}
				//else
					//System.out.println("     NO");
			}
			
			if(instDistVal.size()!=0)
			{
				//Extraires les exemples associés a cet attribut
				ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
				ArrayList <String> FERM_att = new ArrayList <String> ();	
		
				//Liste des instances verifiant la fermeture
				for(int i=0; i<inst.numInstances(); i++)
				{
					if( inst.instance(i).stringValue(m_AttributeSymmetrical.index()) == m_AttributeSymmetrical.value(indDistVal))
						FERM_exe.add(i);
				}
				int nbrInstFer = FERM_exe.size();
				
				if(m_Debug)
				{	
					System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
					for(int i=0; i<nbrInstFer; i++)
						System.out.print (" - "+FERM_exe.get(i));
					System.out.println();
					for(int i=0; i<nbrInstFer; i++)
						System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
				}
				
				//Liste des attributs associés à la fermeture ??????????????????????
				//System.out.println("Extraction des attributs associés à cette fermeture");
				String nl= "-null-";
				for(int i=0; i< (int) inst.numAttributes()-1;i++)
				{
					int cmpt=0;
					String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
					//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
					for (int j=0; j<nbrInstFer;)
					{
						if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
						{
							cmpt++;
							if(cmpt==nbrInstFer)
							{
								FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
								//System.out.println(" ---> ok ");
							}
							j++;
						}
						else
						{
							j=nbrInstFer;
							FERM_att.add(nl); 
							//System.out.println(" ---> null ");
						}
					}
				}
				
				if(m_Debug)
				{
				System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
				for(int i=0; i<FERM_att.size(); i++)
					System.out.print (FERM_att.get(i)+" , ");
				}
				
				
				//Extraire la classe majoritaire associée//		
				int [] nbClasse = new int[inst.numClasses()];			
				for(int k=0;k<inst.numClasses();k++)
					nbClasse[k]=0;
					
				//Parcourir les exemples associée à ce concept
				//System.out.println();
				for(int j=0;j<nbrInstFer;j++)
					nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
					
					
				//Detertminer l'indice de la classe associé
				int indiceMax=0;
				for(int i=0;i<inst.numClasses();i++)
					if(nbClasse[i]>nbClasse[indiceMax])
						indiceMax=i;
				
				if(m_Debug)	{
					System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
					System.out.println ("Liste des des attribut de la fermeture");
					for (int o=0;o<FERM_att.size();o++)
						System.out.print (FERM_att.get(o)+" , ");
					System.out.println ("");
				}
						
				//On retourne le concept Pertinent comme un vecteur 
				ArrayList <String> CP= new ArrayList <String>();
				for (int i=0; i< (int) inst.numAttributes()-1;i++)
						CP.add(FERM_att.get(i));
				
				Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
				
				classifierNC.add(r);
				}
		}
		
  }
 */
	    
	    /*
  if(critere == 8) 
  {
  	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
  	
		for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_AttributeClassifierAttributeEval.index()); indDistVal++)
		{
			instDistVal.clear();
			
			for(int j=0; j<inst.numInstances(); j++)
			{
				//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
				if( inst.instance(j).stringValue(m_AttributeClassifierAttributeEval.index()) == inst.attribute(m_AttributeClassifierAttributeEval.index()).value(indDistVal))
				{
					instDistVal.add(j);
				//	System.out.println("     OK");					
				}
				//else
					//System.out.println("     NO");
			}
			
			if(instDistVal.size()!=0)
			{
				//Extraires les exemples associés a cet attribut
				ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
				ArrayList <String> FERM_att = new ArrayList <String> ();	
		
				//Liste des instances verifiant la fermeture
				for(int i=0; i<inst.numInstances(); i++)
				{
					if( inst.instance(i).stringValue(m_AttributeClassifierAttributeEval.index()) == m_AttributeClassifierAttributeEval.value(indDistVal))
						FERM_exe.add(i);
				}
				int nbrInstFer = FERM_exe.size();
				
				if(m_Debug)
				{	
					System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
					for(int i=0; i<nbrInstFer; i++)
						System.out.print (" - "+FERM_exe.get(i));
					System.out.println();
					for(int i=0; i<nbrInstFer; i++)
						System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
				}
				
				//Liste des attributs associés à la fermeture ??????????????????????
				//System.out.println("Extraction des attributs associés à cette fermeture");
				String nl= "-null-";
				for(int i=0; i< (int) inst.numAttributes()-1;i++)
				{
					int cmpt=0;
					String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
					//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
					for (int j=0; j<nbrInstFer;)
					{
						if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
						{
							cmpt++;
							if(cmpt==nbrInstFer)
							{
								FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
								//System.out.println(" ---> ok ");
							}
							j++;
						}
						else
						{
							j=nbrInstFer;
							FERM_att.add(nl); 
							//System.out.println(" ---> null ");
						}
					}
				}
				
				if(m_Debug)
				{
				System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
				for(int i=0; i<FERM_att.size(); i++)
					System.out.print (FERM_att.get(i)+" , ");
				}
				
				
				//Extraire la classe majoritaire associée//		
				int [] nbClasse = new int[inst.numClasses()];			
				for(int k=0;k<inst.numClasses();k++)
					nbClasse[k]=0;
					
				//Parcourir les exemples associée à ce concept
				//System.out.println();
				for(int j=0;j<nbrInstFer;j++)
					nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
					
					
				//Detertminer l'indice de la classe associé
				int indiceMax=0;
				for(int i=0;i<inst.numClasses();i++)
					if(nbClasse[i]>nbClasse[indiceMax])
						indiceMax=i;
				
				if(m_Debug)	{
					System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
					System.out.println ("Liste des des attribut de la fermeture");
					for (int o=0;o<FERM_att.size();o++)
						System.out.print (FERM_att.get(o)+" , ");
					System.out.println ("");
				}
						
				//On retourne le concept Pertinent comme un vecteur 
				ArrayList <String> CP= new ArrayList <String>();
				for (int i=0; i< (int) inst.numAttributes()-1;i++)
						CP.add(FERM_att.get(i));
				
				Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
				
				classifierNC.add(r);
				}
		}
		
  }
  
  */
	    /*
  if(critere == 9) 
  {
  	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
  	
		for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_AttributeReliefFAttributeEval.index()); indDistVal++)
		{
			instDistVal.clear();
			
			for(int j=0; j<inst.numInstances(); j++)
			{
				//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
				if( inst.instance(j).stringValue(m_AttributeReliefFAttributeEval.index()) == inst.attribute(m_AttributeReliefFAttributeEval.index()).value(indDistVal))
				{
					instDistVal.add(j);
				//	System.out.println("     OK");					
				}
				//else
					//System.out.println("     NO");
			}
			
			if(instDistVal.size()!=0)
			{
				//Extraires les exemples associés a cet attribut
				ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
				ArrayList <String> FERM_att = new ArrayList <String> ();	
		
				//Liste des instances verifiant la fermeture
				for(int i=0; i<inst.numInstances(); i++)
				{
					if( inst.instance(i).stringValue(m_AttributeReliefFAttributeEval.index()) == m_AttributeReliefFAttributeEval.value(indDistVal))
						FERM_exe.add(i);
				}
				int nbrInstFer = FERM_exe.size();
				
				if(m_Debug)
				{	
					System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
					for(int i=0; i<nbrInstFer; i++)
						System.out.print (" - "+FERM_exe.get(i));
					System.out.println();
					for(int i=0; i<nbrInstFer; i++)
						System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
				}
				
				//Liste des attributs associés à la fermeture ??????????????????????
				//System.out.println("Extraction des attributs associés à cette fermeture");
				String nl= "-null-";
				for(int i=0; i< (int) inst.numAttributes()-1;i++)
				{
					int cmpt=0;
					String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
					//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
					for (int j=0; j<nbrInstFer;)
					{
						if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
						{
							cmpt++;
							if(cmpt==nbrInstFer)
							{
								FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
								//System.out.println(" ---> ok ");
							}
							j++;
						}
						else
						{
							j=nbrInstFer;
							FERM_att.add(nl); 
							//System.out.println(" ---> null ");
						}
					}
				}
				
				if(m_Debug)
				{
				System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
				for(int i=0; i<FERM_att.size(); i++)
					System.out.print (FERM_att.get(i)+" , ");
				}
				
				
				//Extraire la classe majoritaire associée//		
				int [] nbClasse = new int[inst.numClasses()];			
				for(int k=0;k<inst.numClasses();k++)
					nbClasse[k]=0;
					
				//Parcourir les exemples associée à ce concept
				//System.out.println();
				for(int j=0;j<nbrInstFer;j++)
					nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
					
					
				//Detertminer l'indice de la classe associé
				int indiceMax=0;
				for(int i=0;i<inst.numClasses();i++)
					if(nbClasse[i]>nbClasse[indiceMax])
						indiceMax=i;
				
				if(m_Debug)	{
					System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
					System.out.println ("Liste des des attribut de la fermeture");
					for (int o=0;o<FERM_att.size();o++)
						System.out.print (FERM_att.get(o)+" , ");
					System.out.println ("");
				}
						
				//On retourne le concept Pertinent comme un vecteur 
				ArrayList <String> CP= new ArrayList <String>();
				for (int i=0; i< (int) inst.numAttributes()-1;i++)
						CP.add(FERM_att.get(i));
				
				Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
				
				classifierNC.add(r);
				}
		}
		
  }
*/
	    
	    /*
  if(critere == 10) 
  {
  	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
  	
		for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_AttributePrincipalComponents.index()); indDistVal++)
		{
			instDistVal.clear();
			
			for(int j=0; j<inst.numInstances(); j++)
			{
				//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
				if( inst.instance(j).stringValue(m_AttributePrincipalComponents.index()) == inst.attribute(m_AttributePrincipalComponents.index()).value(indDistVal))
				{
					instDistVal.add(j);
				//	System.out.println("     OK");					
				}
				//else
					//System.out.println("     NO");
			}
			
			if(instDistVal.size()!=0)
			{
				//Extraires les exemples associés a cet attribut
				ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
				ArrayList <String> FERM_att = new ArrayList <String> ();	
		
				//Liste des instances verifiant la fermeture
				for(int i=0; i<inst.numInstances(); i++)
				{
					if( inst.instance(i).stringValue(m_AttributePrincipalComponents.index()) == m_AttributePrincipalComponents.value(indDistVal))
						FERM_exe.add(i);
				}
				int nbrInstFer = FERM_exe.size();
				
				if(m_Debug)
				{	
					System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
					for(int i=0; i<nbrInstFer; i++)
						System.out.print (" - "+FERM_exe.get(i));
					System.out.println();
					for(int i=0; i<nbrInstFer; i++)
						System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
				}
				
				//Liste des attributs associés à la fermeture ??????????????????????
				//System.out.println("Extraction des attributs associés à cette fermeture");
				String nl= "-null-";
				for(int i=0; i< (int) inst.numAttributes()-1;i++)
				{
					int cmpt=0;
					String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
					//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
					for (int j=0; j<nbrInstFer;)
					{
						if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
						{
							cmpt++;
							if(cmpt==nbrInstFer)
							{
								FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
								//System.out.println(" ---> ok ");
							}
							j++;
						}
						else
						{
							j=nbrInstFer;
							FERM_att.add(nl); 
							//System.out.println(" ---> null ");
						}
					}
				}
				
				if(m_Debug)
				{
				System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
				for(int i=0; i<FERM_att.size(); i++)
					System.out.print (FERM_att.get(i)+" , ");
				}
				
				
				//Extraire la classe majoritaire associée//		
				int [] nbClasse = new int[inst.numClasses()];			
				for(int k=0;k<inst.numClasses();k++)
					nbClasse[k]=0;
					
				//Parcourir les exemples associée à ce concept
				//System.out.println();
				for(int j=0;j<nbrInstFer;j++)
					nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
					
					
				//Detertminer l'indice de la classe associé
				int indiceMax=0;
				for(int i=0;i<inst.numClasses();i++)
					if(nbClasse[i]>nbClasse[indiceMax])
						indiceMax=i;
				
				if(m_Debug)	{
					System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
					System.out.println ("Liste des des attribut de la fermeture");
					for (int o=0;o<FERM_att.size();o++)
						System.out.print (FERM_att.get(o)+" , ");
					System.out.println ("");
				}
						
				//On retourne le concept Pertinent comme un vecteur 
				ArrayList <String> CP= new ArrayList <String>();
				for (int i=0; i< (int) inst.numAttributes()-1;i++)
						CP.add(FERM_att.get(i));
				
				Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
				
				classifierNC.add(r);
				}
		}
		
  }
  
  */
	    
	    /*
  if(critere == 11) //INFORMATION MUTUELLE
  {
  	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
  	
		for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_AttributeInformationMutuelle.index()); indDistVal++)
		{
			instDistVal.clear();
			
			for(int j=0; j<inst.numInstances(); j++)
			{
				//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
				if( inst.instance(j).stringValue(m_AttributeInformationMutuelle.index()) == inst.attribute(m_AttributeInformationMutuelle.index()).value(indDistVal))
				{
					instDistVal.add(j);
					//System.out.println("     OK");					
				}
				//else
					//System.out.println("     NO");
			}
			
			if(instDistVal.size()!=0)
			{
				//Extraires les exemples associés a cet attribut
				ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
				ArrayList <String> FERM_att = new ArrayList <String> ();	
		
				//Liste des instances verifiant la fermeture
				for(int i=0; i<inst.numInstances(); i++)
				{
					if( inst.instance(i).stringValue(m_AttributeInformationMutuelle.index()) == m_AttributeInformationMutuelle.value(indDistVal))
						FERM_exe.add(i);
				}
				int nbrInstFer = FERM_exe.size();
				
				if(m_Debug)
				{	
					System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
					for(int i=0; i<nbrInstFer; i++)
						System.out.print (" - "+FERM_exe.get(i));
					System.out.println();
					for(int i=0; i<nbrInstFer; i++)
						System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
				}
				
				//Liste des attributs associés à la fermeture ??????????????????????
				////System.out.println("Extraction des attributs associés à cette fermeture");
				String nl= "-null-";
				for(int i=0; i< (int) inst.numAttributes()-1;i++)
				{
					int cmpt=0;
					String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
					System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
					for (int j=0; j<nbrInstFer;)
					{
						if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
						{
							cmpt++;
							if(cmpt==nbrInstFer)
							{
									
								FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
								////System.out.println(" ---> ok ");
							}
							j++;
						}
						else
						{
							j=nbrInstFer;
							FERM_att.add(nl); 
							////System.out.println(" ---> null ");
						}
					}
				}
				
				if(m_Debug)
				{
				System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
				for(int i=0; i<FERM_att.size(); i++)
					System.out.print (FERM_att.get(i)+" , ");
				}
				
				
				//Extraire la classe majoritaire associée//		
						
				int [] nbClasse = new int[inst.numClasses()];			
				for(int k=0;k<inst.numClasses();k++)
					nbClasse[k]=0;
					
				//Parcourir les exemples associée à ce concept
				////System.out.println();
				for(int j=0;j<nbrInstFer;j++)
					nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
					
					
				//Detertminer l'indice de la classe associé
				int indiceMax=0;
				for(int i=0;i<inst.numClasses();i++)
					if(nbClasse[i]>nbClasse[indiceMax])
						indiceMax=i;
				
				if(m_Debug)	{
					System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
					System.out.println ("Liste des des attribut de la fermeture");
					for (int o=0;o<FERM_att.size();o++)
						System.out.print (FERM_att.get(o)+" , ");
					System.out.println ("");
				}
						
				//On retourne le concept Pertinent comme un vecteur 
				ArrayList <String> CP= new ArrayList <String>();
				for (int i=0; i< (int) inst.numAttributes()-1;i++)
						CP.add(FERM_att.get(i));
				
				Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
				
				classifierNC.add(r);
				}
		}
		
  }
  
  */
	    
	    /*
  if(critere == 12) 
  {
  	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
  	
		for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_AttributeCorreRatio.index()); indDistVal++)
		{
			instDistVal.clear();
			
			for(int j=0; j<inst.numInstances(); j++)
			{
				//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
				if( inst.instance(j).stringValue(m_AttributeCorreRatio.index()) == inst.attribute(m_AttributeCorreRatio.index()).value(indDistVal))
				{
					instDistVal.add(j);
					//System.out.println("     OK");					
				}
				//else
					//System.out.println("     NO");
			}
			
			if(instDistVal.size()!=0)
			{
				//Extraires les exemples associés a cet attribut
				ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
				ArrayList <String> FERM_att = new ArrayList <String> ();	
		
				//Liste des instances verifiant la fermeture
				for(int i=0; i<inst.numInstances(); i++)
				{
					if( inst.instance(i).stringValue(m_AttributeCorreRatio.index()) == m_AttributeCorreRatio.value(indDistVal))
						FERM_exe.add(i);
				}
				int nbrInstFer = FERM_exe.size();
				
				if(m_Debug)
				{	
					System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
					for(int i=0; i<nbrInstFer; i++)
						System.out.print (" - "+FERM_exe.get(i));
					System.out.println();
					for(int i=0; i<nbrInstFer; i++)
						System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
				}
				
				//Liste des attributs associés à la fermeture ??????????????????????
				////System.out.println("Extraction des attributs associés à cette fermeture");
				String nl= "-null-";
				for(int i=0; i< (int) inst.numAttributes()-1;i++)
				{
					int cmpt=0;
					String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
					System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
					for (int j=0; j<nbrInstFer;)
					{
						if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
						{
							cmpt++;
							if(cmpt==nbrInstFer)
							{
									
								FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
								////System.out.println(" ---> ok ");
							}
							j++;
						}
						else
						{
							j=nbrInstFer;
							FERM_att.add(nl); 
							////System.out.println(" ---> null ");
						}
					}
				}
				
				if(m_Debug)
				{
				System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
				for(int i=0; i<FERM_att.size(); i++)
					System.out.print (FERM_att.get(i)+" , ");
				}
				
				
				//Extraire la classe majoritaire associée//		
						
				int [] nbClasse = new int[inst.numClasses()];			
				for(int k=0;k<inst.numClasses();k++)
					nbClasse[k]=0;
					
				//Parcourir les exemples associée à ce concept
				////System.out.println();
				for(int j=0;j<nbrInstFer;j++)
					nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
					
					
				//Detertminer l'indice de la classe associé
				int indiceMax=0;
				for(int i=0;i<inst.numClasses();i++)
					if(nbClasse[i]>nbClasse[indiceMax])
						indiceMax=i;
				
				if(m_Debug)	{
					System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
					System.out.println ("Liste des des attribut de la fermeture");
					for (int o=0;o<FERM_att.size();o++)
						System.out.print (FERM_att.get(o)+" , ");
					System.out.println ("");
				}
						
				//On retourne le concept Pertinent comme un vecteur 
				ArrayList <String> CP= new ArrayList <String>();
				for (int i=0; i< (int) inst.numAttributes()-1;i++)
						CP.add(FERM_att.get(i));
				
				Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
				
				classifierNC.add(r);
				}
		}
		
  }
  
  */
	    
	    /*
 
  if(critere == 13) // H-Ratio
  {
  	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
  	
		for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_AttributeHRatio.index()); indDistVal++)
		{
			instDistVal.clear();
			
			for(int j=0; j<inst.numInstances(); j++)
			{
				//System.out.print((j+1)+"iéme instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
				if( inst.instance(j).stringValue(m_AttributeHRatio.index()) == inst.attribute(m_AttributeHRatio.index()).value(indDistVal))
				{
					instDistVal.add(j);
					//System.out.println("     OK");					
				}
				//else
					//System.out.println("     NO");
			}
			
			if(instDistVal.size()!=0)
			{
				//Extraires les exemples associés a cet attribut
				ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
				ArrayList <String> FERM_att = new ArrayList <String> ();	
		
				//Liste des instances verifiant la fermeture
				for(int i=0; i<inst.numInstances(); i++)
				{
					if( inst.instance(i).stringValue(m_AttributeHRatio.index()) == m_AttributeHRatio.value(indDistVal))
						FERM_exe.add(i);
				}
				int nbrInstFer = FERM_exe.size();
				
				if(m_Debug)
				{	
					System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
					for(int i=0; i<nbrInstFer; i++)
						System.out.print (" - "+FERM_exe.get(i));
					System.out.println();
					for(int i=0; i<nbrInstFer; i++)
						System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
				}
				 
				//Liste des attributs associés à la fermeture ??????????????????????
				////System.out.println("Extraction des attributs associés à cette fermeture");
				String nl= "-null-";
				for(int i=0; i< (int) inst.numAttributes()-1;i++)
				{
					int cmpt=0;
					String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
					System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
					for (int j=0; j<nbrInstFer;)
					{
						if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
						{
							cmpt++;
							if(cmpt==nbrInstFer)
							{
									
								FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
								////System.out.println(" ---> ok ");
							}
							j++;
						}
						else
						{
							j=nbrInstFer;
							FERM_att.add(nl); 
							////System.out.println(" ---> null ");
						}
					}
				}
				
				if(m_Debug)
				{
				System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
				for(int i=0; i<FERM_att.size(); i++)
					System.out.print (FERM_att.get(i)+" , ");
				}
				
				
				//Extraire la classe majoritaire associée//		
						
				int [] nbClasse = new int[inst.numClasses()];			
				for(int k=0;k<inst.numClasses();k++)
					nbClasse[k]=0;
					
				//Parcourir les exemples associée à ce concept
				////System.out.println();
				for(int j=0;j<nbrInstFer;j++)
					nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
					
					
				//Detertminer l'indice de la classe associé
				int indiceMax=0;
				for(int i=0;i<inst.numClasses();i++)
					if(nbClasse[i]>nbClasse[indiceMax])
						indiceMax=i;
				
				if(m_Debug)	{
					System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
					System.out.println ("Liste des des attribut de la fermeture");
					for (int o=0;o<FERM_att.size();o++)
						System.out.print (FERM_att.get(o)+" , ");
					System.out.println ("");
				}
						
				//On retourne le concept Pertinent comme un vecteur 
				ArrayList <String> CP= new ArrayList <String>();
				for (int i=0; i< (int) inst.numAttributes()-1;i++)
						CP.add(FERM_att.get(i));
				
				Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
				
				classifierNC.add(r);
				}
		}
		
  }
  
  */

  return classifierNC;
  }
 
  

	/**
   * Computes information gain for an attribute.
   *
   * @param data the data for which info gain is to be computed
   * @param att the attribute
   * @return the information gain for the given attribute and data
   * @throws Exception if computation fails
   */
  private double computeInfoGain(Instances data, Attribute att) throws Exception {

    double infoGain = computeEntropy(data);
    //System.out.println("\nEntropy data: " + computeEntropy(data) + "\t att: "+ att.name() +"\t InfoGain Initial: "+ infoGain); 
    Instances[] splitData = splitData(data, att);
    for (int j = 0; j < att.numValues(); j++) 
    {
      if (splitData[j].numInstances() > 0) 
      {
        infoGain -= ((double) splitData[j].numInstances() / (double) data.numInstances()) * computeEntropy(splitData[j]);
        /*System.out.println("\t att.Values():" + att.value(j) + "\t infoGain - : " + splitData[j].numInstances() 
        		+ computeEntropy(splitData[j]) + "*" + splitData[j].numInstances() + "/" + data.numInstances() + 
        				" = " + infoGain
        				);*/ 
      } 
    }
    //System.out.println("=====> attribut: "+ att.name() +"\t Final InfoGain: "+ infoGain); 
    
    return infoGain;
  }

  /**
   * Computes the entropy of a dataset.
   * 
   * @param data the data for which entropy is to be computed
   * @return the entropy of the data's class distribution
   * @throws Exception if computation fails
   */
  private double computeEntropy(Instances data) throws Exception {

    double [] classCounts = new double[data.numClasses()];
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      classCounts[(int) inst.classValue()]++;
    }
    double entropy = 0;
    for (int j = 0; j < data.numClasses(); j++) {
      if (classCounts[j] > 0) {
        entropy -= classCounts[j] * Utils.log2(classCounts[j]);
      }
    }
    entropy /= (double) data.numInstances();
    return entropy + Utils.log2(data.numInstances());
  }

  /**
   * Splits a dataset according to the values of a nominal attribute.
   *
   * @param data the data which is to be split
   * @param att the attribute to be used for splitting
   * @return the sets of instances produced by the split
   */
  private Instances[] splitData(Instances data, Attribute att) throws Exception {

	    Instances[] splitData = new Instances[att.numValues()];
	    for (int j = 0; j < att.numValues(); j++) {
	      splitData[j] = new Instances(data, data.numInstances());
	    }
	    Enumeration instEnum = data.enumerateInstances();
	    while (instEnum.hasMoreElements()) {
	      Instance inst = (Instance) instEnum.nextElement();
	      splitData[(int) inst.value(att)].add(inst);
	    }
	    for (int i = 0; i < splitData.length; i++) {
	      splitData[i].compactify();
	    }
	    return splitData;
	  }

     
	 	 
  //public String toString() 
 // {
	//StringBuffer sb = new StringBuffer("[Classifier Nominal Concept Notes]\n"); 
	//calendar = Calendar.getInstance();
	
	//if((m_CNC==1)&&(this.m_ZeroR==null))
	//{	
		//sb.append("[CNC] \t Running OK ... \n"); 
		//sb.append("[CNC] \t The next display can be missing ... \n"); 
		//for(int i=0; i<m_classifierNC.size();i++)
			//sb.append("[CNC] \t ("+sdf.format(calendar.getTime())+") Classifier "+(i+1)+" : "+m_classifierNC.get(i).affich_nom_rule(true) + "\n");
	//	if(m_Debug)
			//sb.append("[CNC] \t A Log Console is generated... \n");
		//return sb.toString();
	//}
	//else
		//return "[CNC] \t m_ZeroR model \n" ;
//}
  
  
  


@Override
public String toSource(String className) throws Exception {
	// TODO Auto-generated method stub
	return null;
}
	
/** DecisionStump.java
 * Computes variance for subsets.
 * 
 * @param s
 * @param sS
 * @param sumOfWeights
 * @return the variance
 */
protected double variance(double[][] s,double[] sS,double[] sumOfWeights) {

  double var = 0;

  for (int i = 0; i < s.length; i++) {
    if (sumOfWeights[i] > 0) {
	var += sS[i] - ((s[i][0] * s[i][0]) / (double) sumOfWeights[i]);
    }
  }
  
  return var;
}

/** Decision Stumps.java
 * Returns the value as string out of the given distribution
 * 
 * @param c the attribute to get the value for
 * @param dist the distribution to extract the value
 * @return the value
 */
protected String sourceClass(Attribute c, double []dist) {

  if (c.isNominal()) {
    return Integer.toString(Utils.maxIndex(dist));
  } else {
    return Double.toString(dist[0]);
  }
}

/**DecisionStump.java
 * Calculates the class membership probabilities for the given test instance.
 *
 * @param instance the instance to be classified
 * @return predicted class probability distribution
 * @throws Exception if distribution can't be computed
 */
/*public double[] distributionForInstance(Instance instance) throws Exception {

  // default model?
  if (m_ZeroR != null) {
    return m_ZeroR.distributionForInstance(instance);
  }
  
  return m_Distribution[whichSubset(instance)];
}*/













  
/**
 * Main method for testing this class.
 *
 * @param argv the options
 */
public static void main(String [] argv) {
  runClassifier(new CNC(), argv);
}

@Override
public void updateClassifier(Instance instance) throws Exception {
	// TODO Auto-generated method stub
	System.out.println("/** updateClassifier(Instance instance) **/");  
	
}


}