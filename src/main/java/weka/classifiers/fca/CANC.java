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
 *    CANC.java
 *    Copyright (C) 2016 Laboratory of computing, Programming, Algorithmic and Heuristic (LIPAH),
 *	  Faculty of Mathematical, Physical and Natural Sciences of Tunis (FST),
 *	  El Manar University, 1060, Tunis, Tunisia.
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
import weka.core.Instance;
import weka.core.Instances;
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

// import JFlex.Out;

/**
<!-- globalinfo-start -->
* Class for building and using a Classifier Nominal Concept. 
* Usually used in conjunction with a boosting/bagging algorithm. 
* Does classification (based on entropy, gain ...). 
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
* @version $Revision: 160816 $
*/

//public class CNC extends Classifier 
//implements UpdateableClassifier, OptionHandler, TechnicalInformationHandler 


public class CANC 
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
	protected double [][] m_Distribution;
	
	/** The instances used for training. */
	protected Instances m_Instances;
	
	/** The filter used to get rid of missing values. */
	protected ReplaceMissingValues m_Missing = new ReplaceMissingValues();
	  
	/** a ZeroR model in case no model can be built from the data */
	protected Classifier m_ZeroR;	
	
	/** Percent of minimum plurality vote*/
	protected double m_PlurMin = (double) 66.66; 
	
	/** Percent of low personel ponderate vote*/
	protected double m_LowPondVoter = (double) 66.66; 
	
	/** Number of best ponderate voters */
	protected int m_NumBestVoterS = (int) 3; 
    
    // Une structure contenant les régles uniques
    public static ArrayList <Classification_Rule> m_classifierNC ;
    
    protected static InfoGainAttributeEval m_InfoGainAttributeEval = new InfoGainAttributeEval();
	
    
    /**
	 * Data Set Specification (before discretization)
	 */
    
    public static final int DSS_no = 0;  		// Default
    public static final int DSS_full = 1;  		// 
    public static final int DSS_light = 2;  	// Experimental
    
    private int DataSetSpecif = 0; // Experimental
    
    public static final Tag [] TAGS_DataSetSpecif = {
    		new Tag(DSS_no, "No data set specification"),
    		new Tag(DSS_full, "Full data set specification"),
    		new Tag(DSS_light, "Minimum data set specification"),
        	};
    
    public SelectedTag getDataSetSpecif() {	
    	return new SelectedTag(DataSetSpecif, TAGS_DataSetSpecif);	
    	}
    
    public void setDataSetSpecif(SelectedTag agregation) {
    	if (agregation.getTags() == TAGS_DataSetSpecif)
    		this.DataSetSpecif = agregation.getSelectedTag().getID();
    	}

    /**
	 * Data Set Specification (after discretization) 
	 */
    
    public static final int DSS_Disc_no = 0;  			// Default
    public static final int DSS_Disc_full = 1;  		// 
    public static final int DSS_Disc_light = 2;  		// Experimental
    
    private int DataSetSpecif_Discretized = 0; 		// Experimental
    
    public static final Tag [] TAGS_DataSetSpecif_Discretized = {
    		new Tag(DSS_Disc_no, "No data set discretized specification"),
    		new Tag(DSS_Disc_full, "Full data set discretized specification"),
    		new Tag(DSS_Disc_light, "Minimum data set discretized specification"),
        	};
    
    public SelectedTag getDataSetSpecif_Discretized() {	
    	return new SelectedTag(DataSetSpecif_Discretized, TAGS_DataSetSpecif_Discretized);	
    	}
    
    public void setDataSetSpecif_Discretized(SelectedTag agregation) {
    	if (agregation.getTags() == TAGS_DataSetSpecif_Discretized)
    		this.DataSetSpecif_Discretized = agregation.getSelectedTag().getID();
    	}

    /**
	 * L'apprentissage du concept nominal : 
	 * Soit la fermeture du meilleur(e) attribut/valeur 
	 * ou la fermeture de chaque attribut/valeur 
	 */
    
    public static final int CONCEPT_LEARNING_FMAN = 1;  // Default: Fermeture du Meilleur Attribut Nominal
    public static final int CONCEPT_LEARNING_FTAN = 2;  // Experimental: Fermeture de TOUT les Attributs Nominaux
    
    private int NominalConceptLearning = CONCEPT_LEARNING_FTAN; // Experimental
    
    public static final Tag [] TAGS_NominalConceptLearning = {
    		new Tag(CONCEPT_LEARNING_FMAN, "Closure of the best nominal attribut"),
    		new Tag(CONCEPT_LEARNING_FTAN, "Closure of ALL nominal attributs"),
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
    
    private int FMANmeasure = FMAN_GAIN_INFO_BV;
    
    public static final Tag [] TAGS_FMANmeasure = {
		new Tag(FMAN_GAIN_INFO_BV, "Info. Gain & Best Value"),
		new Tag(FMAN_GAIN_INFO_BA, "Info. Gain & Best Attributt"),
		new Tag(FMAN_GAIN_INFO_MV, "Info. Gain & Multi Values "),
		};
    
    public SelectedTag getFMAN_Measure() {
		return new SelectedTag(FMANmeasure, TAGS_FMANmeasure);
		}

	public void setFMAN_Measure(SelectedTag agregation) {
		if (agregation.getTags() == TAGS_FMANmeasure)
			this.FMANmeasure = agregation.getSelectedTag().getID();
		}

    /**
     * Fermeture du Meilleur Attribut Nominal : choix de(s) valeur(s) nominale(s)
     */
    
    public static final int FTAN_BV = 1;	// Default: La valeur la plus pertinente (support) de l'attribut selectionné
    public static final int FTAN_MV = 2;	// Les valeurs nominales de l'attribut selectionné
    
    private int FTANmeasure = FTAN_BV;
    
    public static final Tag [] TAGS_FTANmeasure = {
		new Tag(FTAN_BV, "Best Value"),
		new Tag(FTAN_MV, "Multi Values "),
		};
    
    public SelectedTag getFTAN_Measure() {
		return new SelectedTag(FTANmeasure, TAGS_FTANmeasure);
		}

	public void setFTAN_Measure(SelectedTag agregation) {
		if (agregation.getTags() == TAGS_FTANmeasure)
			this.FTANmeasure = agregation.getSelectedTag().getID();
		}
	
	/**
     * Le choix de la technique du vote 
     * en cas où nous avons retenu tout les valeurs nominales
     * de l'attribut qui maximise le gain Informationel.
     * ou bien
     * en cas ou nous avons retenu TOUT les attributs nominaux
     */
    
    public static final int Vote_Pond = 1;				// Vote pondéré
    public static final int Vote_Maj = 2;				// Default: Vote majoritaire
    public static final int Vote_Plur = 3;				// Vote à pluralité (supérieur à 50%+1 des votes)
    public static final int Vote_PlurMin = 4;			// Vote à pluralité avec un seuil minimum (%)
    public static final int Vote_NumBestVoterS = 5;		// Vote pondéré les meilleurs classifieurs (High Ponderation)
    public static final int Vote_LowPondVoter = 6;		// Vote pondéré des classifieurs qui maximisent un seuil donné
    
    private int VoteMethods = 6; // Experimental
    
    public static final Tag [] TAGS_VoteMethods = {
		new Tag(Vote_Pond, "Ponderate Vote"),
		new Tag(Vote_Maj, "Majority Vote"),
		new Tag(Vote_Plur, "Plurality Vote (50%+1)"),
		new Tag(Vote_PlurMin, "Plurality Vote Minimum (arg:PlurMin)"),
		new Tag(Vote_NumBestVoterS, "Best Voters (arg:numBestVoterS)"),
		new Tag(Vote_LowPondVoter, "Minimum ponderation of Voters (arg:lowPondVoter)"),
		};
    
    public SelectedTag getVote_Methods() {
		return new SelectedTag(VoteMethods, TAGS_VoteMethods);
		}

	public void setVote_Methods(SelectedTag agregation) {
		if (agregation.getTags() == TAGS_VoteMethods)
			this.VoteMethods = agregation.getSelectedTag().getID();
		}

	/**
	   * Gets the percent of minimum plurality vote.
	   *
	   * @return the percent 
	   */
	  public double getPlurMin() {
	    return m_PlurMin;
	  }
	  
	  /**
	   * Sets the percent of minimum plurality vote.
	   *
	   * @param value     the new percent
	   */
	  public void setPlurMin(double value) {
	    if (value >= 0)
	    	m_PlurMin = value;
	    else
	      System.out.println(
	          "Percent should be superior or equal to zero (provided: " + value + ")!");
	  }
	  

	/**
	   * Gets the percent of minimum plurality vote.
	   *
	   * @return the percent 
	   */
	  public double getLowPondVoter() {
	    return m_LowPondVoter;
	  }
	  
	  /**
	   * Sets the percent of minimum ponderation of a voter.
	   *
	   * @param value     the new percent
	   */
	  public void setLowPondVoter(double value) {
	    if (value >= 0)
	    	m_LowPondVoter = value;
	    else
	      System.out.println(
	          "Percent should be superior or equal to zero (provided: " + value + ")!");
	  }
		  
	  /**
	   * Gets the maximum of best voters.
	   *
	   * @return the number (int) 
	   */
	  public int getNumBestVoterS() {
	    return m_NumBestVoterS;
	  }
	  
	  /**
	   * Sets the maximum of best voters.
	   *
	   * @param value     the new number
	   */
	  public void setNumBestVoterS(int value) {
	    if (value >= 1)
	    	m_NumBestVoterS = value;
	    else
	      System.out.println(
	          "Number should be strict superior to zeo (provided: " + value + ")!");
	  }

	/**
	 * Un filtre permettant de transformer les données numériques en données nominales
	 */

	//protected static Filter m_Filter = new weka.filters.unsupervised.attribute.Discretize();
	protected static Filter m_Filter = new weka.filters.supervised.attribute.Discretize();
	
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
				  "\t Full class name of filter to use, followed\n"
						  + "\t by filter options.\n"
//						  + "\t eg: \"weka.filters.unsupervised.attribute.Remove -V -R 1,2\"",
						  + "\t eg: \"weka.filters.supervised.attribute.Remove -V -R 1,2\"",
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
	   * <pre> -Q 											// A re faire
	   *  Use resampling for boosting.</pre>				// A re faire
	   * 
	   * <pre> -I &lt;num&gt;								// A re faire
	   *  Number of iterations.								// A re faire
	   *  (default 10)</pre>								// A re faire
	   * 
	   * <pre> -D											// A re faire
	   *  If set, classifier is run in debug mode and		// A re faire
	   *  may output additional info to the console</pre>	// A re faire
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
				//NominalConceptLearning = CONCEPT_LEARNING_FTAN; // Default
				NominalConceptLearning = CONCEPT_LEARNING_FTAN; // Experimental
			
			switch (NominalConceptLearning) { 
			case CONCEPT_LEARNING_FMAN:    NominalConceptLearning = 1; break;
			case CONCEPT_LEARNING_FTAN:    NominalConceptLearning = 2; break;
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
							
			switch (FMANmeasure) { 
			 case FMAN_GAIN_INFO_BV:	FMANmeasure = 1; break;
			 case FMAN_GAIN_INFO_BA:	FMANmeasure = 2; break;
			 case FMAN_GAIN_INFO_MV:	FMANmeasure = 3; break;
		     }	  
			
			//Les metriques de la selection du meilleur attribut nominal
			runString = Utils.getFlag("FTAN_BV", options);
			if (runString)
				FTANmeasure = FTAN_BV;
							
			runString = Utils.getFlag("FTAN_MV", options);
			if (runString)
				FTANmeasure = FTAN_MV;
							
			switch (FMANmeasure) { 
			 case FTAN_BV:	FTANmeasure = 1; break;
			 case FTAN_MV:	FTANmeasure = 2; break;
		     }	  
			
			// Les techniques de vote dans le cas de la fermeture des valeurs 
			// nominales de l'attribut qui m'aximise le gain informationel
			runString = Utils.getFlag("Vote_Pond", options);
			if ((FMANmeasure == FMAN_GAIN_INFO_MV) && runString)
				VoteMethods = Vote_Pond;
					
			runString = Utils.getFlag("Vote_Maj", options);
			if ((FMANmeasure == FMAN_GAIN_INFO_MV) && runString)
				VoteMethods = Vote_Maj;
					
			runString = Utils.getFlag("Vote_Plur", options);
			if ((FMANmeasure == FMAN_GAIN_INFO_MV) && runString)
				VoteMethods = Vote_Plur;
			
			runString = Utils.getFlag("Vote_PlurMin", options);
			if ((FMANmeasure == FMAN_GAIN_INFO_MV) && runString)
				VoteMethods = Vote_PlurMin;
					
			runString = Utils.getFlag("Vote_NumBestVoterS", options);
			if ((FMANmeasure == FMAN_GAIN_INFO_MV) && runString)
				VoteMethods = Vote_NumBestVoterS;
			
			runString = Utils.getFlag("Vote_LowPondVoter", options);
			if ((FMANmeasure == FMAN_GAIN_INFO_MV) && runString)
				VoteMethods = Vote_LowPondVoter;
					
			switch (VoteMethods) 
			{ 
				 case Vote_Pond:			VoteMethods = 1; 	break;
				 case Vote_Maj:				VoteMethods = 2; 	break;
				 case Vote_Plur:			VoteMethods = 3; 	break;
				 case Vote_PlurMin:			VoteMethods = 4; 	break;
				 case Vote_NumBestVoterS:	VoteMethods = 5; 	break;
				 case Vote_LowPondVoter:	VoteMethods = 6; 	break;
		     }	 
			
			// Same for filter
		    String filterString = Utils.getOption('F', options);
		    if (filterString.length() > 0) 
		    {
		      String [] filterSpec = Utils.splitOptions(filterString);
		      if (filterSpec.length == 0)
		    	  throw new IllegalArgumentException("Invalid filter specification string");
		      String filterName = filterSpec[0];
		      filterSpec[0] = "";
		      setFilter((Filter) Utils.forName(Filter.class, filterName, filterSpec));
		    } 
		    else 
		      setFilter(new weka.filters.supervised.attribute.Discretize());
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
		  case CONCEPT_LEARNING_FMAN: 	
			  result.add("-fman"); 
			  break;
		  case CONCEPT_LEARNING_FTAN: 	
			  result.add("-ftan"); 
			  switch(VoteMethods) 
			  {
				  case Vote_Pond:			result.add("-pondVote"); 										break;
				  case Vote_Maj:			result.add("-majVote"); 										break; 
				  case Vote_Plur:			result.add("-plurVote"); 										break; 
				  case Vote_PlurMin:		result.add("-plurMinVote"); result.add("-"+m_PlurMin); 			break; 
				  case Vote_NumBestVoterS:	result.add("-NumBestVoterS"); result.add("-"+m_NumBestVoterS); 	break; 
				  case Vote_LowPondVoter:	result.add("-LowPondVoter"); result.add("-"+m_LowPondVoter); 	break; 
			  }
			  break;
		  }

		  if(NominalConceptLearning == CONCEPT_LEARNING_FMAN)
			  switch(FMANmeasure) 
			  {
			  case FMAN_GAIN_INFO_BV:	
				  result.add("-giBestV"); break;			    	
			  case FMAN_GAIN_INFO_BA:	
				  result.add("-giBestA"); break;			    	
			  case FMAN_GAIN_INFO_MV:	
				  result.add("-giMultiV"); 
				  switch(VoteMethods) 
				  {
				  case Vote_Pond:			result.add("-pondVote"); 										break;
				  case Vote_Maj:			result.add("-majVote"); 										break; 
				  case Vote_Plur:			result.add("-plurVote"); 										break; 
				  case Vote_PlurMin:		result.add("-plurMinVote"); result.add("-"+m_PlurMin); 			break; 
				  case Vote_NumBestVoterS:	result.add("-NumBestVoterS"); result.add("-"+m_NumBestVoterS); 	break; 
				  case Vote_LowPondVoter:	result.add("-LowPondVoter"); result.add("-"+m_LowPondVoter); 	break; 
				  }
				  break;
			  }
		  
		  if(NominalConceptLearning == CONCEPT_LEARNING_FTAN)
			  switch(FTANmeasure) 
			  {
			  case FTAN_BV:	result.add("-BestV"); break;		    	
			  case FTAN_MV:	result.add("-MultiV"); break;
			  }
		  
		  if(m_Debug)
				 result.add("-MODE_DEBUG");		// Mode Debug
		  
		  result.add(getFilterSpec());
			    
		  return (String[]) result.toArray(new String[result.size()]);
	  }
		
	  public String generationTipText() { 
		  return "If set to rules, Classifier Nominal Concept may output in the log file all rules generated." ; 
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

    if (m_Filter.numPendingOutput() > 0) 
      throw new Exception("Filter output queue not empty!");
    
    if (!m_Filter.input(inst))
      throw new Exception("Filter didn't make the test instance immediately available!");

    m_Filter.batchFinished();
    Instance newInstance = m_Filter.output();

    //System.err.println("FilteredClassifier:: " + m_Filter.getClass().getName() + " out: " + newInstance); 
    
	m_Missing.input(inst);
	m_Missing.batchFinished();
	inst = m_Missing.output();

	  double result= (double) -1.0;
	  Classify_Instance  listRules = new Classify_Instance();
	  
	  if(NominalConceptLearning == CONCEPT_LEARNING_FMAN)
		  switch(FMANmeasure)
		  {
			  case FMAN_GAIN_INFO_BA : 
				  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
				  break;
			  case FMAN_GAIN_INFO_BV : 
				  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
				  break;
			  case FMAN_GAIN_INFO_MV : 
				  switch(VoteMethods)
				  {
				  case Vote_Pond: 
					  result = (double) listRules.classify_Instance_nom_VotePond(newInstance, m_classifierNC, newInstance.numClasses()); 
					  break;
				  case Vote_Maj: 
					  result = (double) listRules.classify_Instance_nom_VoteMaj(newInstance, m_classifierNC, newInstance.numClasses()); 
					  break;
				  case Vote_Plur: 
					  result = (double) listRules.classify_Instance_nom_VotePlur(newInstance, m_classifierNC, newInstance.numClasses()); 
					  break;
				  case Vote_PlurMin: 
					  result = (double) listRules.classify_Instance_nom_VotePlurMin(newInstance, m_classifierNC, newInstance.numClasses(), m_PlurMin); 
					  break;
				  case Vote_NumBestVoterS: 
					  result = (double) listRules.classify_Instance_nom_VoteNumBestVoterS(newInstance, m_classifierNC, newInstance.numClasses(), m_NumBestVoterS); 
					  break;
				  case Vote_LowPondVoter: 
					  result = (double) listRules.classify_Instance_nom_VoteLowPondVoter(newInstance, m_classifierNC, newInstance.numClasses(), m_LowPondVoter); 
					  break;
				  }
				  break;
		  }
	  else 
		  if(NominalConceptLearning == CONCEPT_LEARNING_FTAN)
			  switch(VoteMethods)
			  {
				  case Vote_Pond: 
					  result = (double) listRules.classify_Instance_nom_VotePond(newInstance, m_classifierNC, newInstance.numClasses()); 
					  break;
				  case Vote_Maj: 
					  result = (double) listRules.classify_Instance_nom_VoteMaj(newInstance, m_classifierNC, newInstance.numClasses()); 
					  break;
				  case Vote_Plur: 
					  result = (double) listRules.classify_Instance_nom_VotePlur(newInstance, m_classifierNC, newInstance.numClasses()); 
					  break;
				  case Vote_PlurMin: 
					  result = (double) listRules.classify_Instance_nom_VotePlurMin(newInstance, m_classifierNC, newInstance.numClasses(), m_PlurMin); 
					  break;
				  case Vote_NumBestVoterS: 
					  result = (double) listRules.classify_Instance_nom_VoteNumBestVoterS(newInstance, m_classifierNC, newInstance.numClasses(), m_NumBestVoterS); 
					  break;
				  case Vote_LowPondVoter: 
					  result = (double) listRules.classify_Instance_nom_VoteLowPondVoter(newInstance, m_classifierNC, newInstance.numClasses(), m_LowPondVoter); 
					  break;
			  }		  
	  
	  if(m_Debug)
	  {
		  System.out.println ("Liste des régles de classification");
		  for (int i=0; i<m_classifierNC.size(); i++)
			  System.out.println (m_classifierNC.get(i).affich_nom_rule(true));
		  System.out.println("L'instance à prédire sa classe: " + newInstance.toString()+"\t --> \t"+result);
		  System.out.println("\n\n\n");
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
  public void buildClassifier(Instances instances) throws Exception 
  {
	  switch(DataSetSpecif)
	  {
	  case 1: DataSetSpecification(instances); break;
	  case 2: DataSetSpecificationLight(instances); break;
	  }
  
	  //Calendar calendar = Calendar.getInstance(); SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS"); System.err.println("\n \t"+sdf.format(calendar.getTime()));
			      
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

	  switch(DataSetSpecif_Discretized)
	  {
	  case 1: DataSetSpecification(m_FilteredInstances); break;
	  case 2: DataSetSpecificationLight(m_FilteredInstances); break;
	  }
	    
	  // only class? -> build ZeroR model
	  if (m_FilteredInstances.numAttributes() == 1) 
	  {
		  System.err.println("Cannot build model (only class attribute present in data!), "
	       + "using ZeroR model instead!");
		  m_ZeroR = new weka.classifiers.rules.ZeroR();
		  m_ZeroR.buildClassifier(m_FilteredInstances);
		  return;
	  } 
	  else 
	  {
	    m_ZeroR = null;
	    this.m_CNC = 1; // build CNC model
	  }
	      
	  
	  switch(this.NominalConceptLearning)
	  {
	  case CONCEPT_LEARNING_FMAN: 
		  if(m_Debug){
			  calendar = Calendar.getInstance();
			  System.err.println("\n t"+sdf.format(calendar.getTime()));  
		  }
		  buildClassifierWithNominalClosure(m_FilteredInstances);  
		  break;
	  case CONCEPT_LEARNING_FTAN: 
		  if(m_Debug){
			  calendar = Calendar.getInstance();
			  System.err.println("\n t"+sdf.format(calendar.getTime()));  
		  }
		  buildClassifierWithAllNominalClosure(m_FilteredInstances);   
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
	  }
	  
	  if(m_Debug)	{
		  System.out.println("\n\n\t=== Vector CLASSIFIER NOMINAL CONCEPT ===");
		  for(int i=0; i<m_classifierNC.size();i++)
			  System.out.println("CNC["+i+"]: "+m_classifierNC.get(i).affich_nom_rule(true));
		}
}  
  

  protected void buildClassifierWithAllNominalClosure(Instances LearningData) throws Exception {

	  //Calendar calendar = Calendar.getInstance(); 
	  //SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS"); 
	  //System.err.println(sdf.format(calendar.getTime())+ "\n");
		
	  m_classifierNC = new ArrayList <Classification_Rule> ();	
	  m_classifierNC.clear();
	  
	  switch (this.FTANmeasure) 
	  {
		  case FTAN_BV: // Fermeture de la valeur nominale la plus pertienente (Support) de l'attribut nominal 
			  m_classifierNC = ExtraireRegleFermAllNom(LearningData, 1);
			  break; 	    
		  case FTAN_MV: // Fermetures des valeurs nominales de l'attribut nominal
			  m_classifierNC = ExtraireRegleFermAllNom(LearningData, 2);
			  break;
	  }
	  
	  if(m_Debug)	
	  {
		  System.out.println("\n\n\t=== Vector CLASSIFIER all NOMINAL CONCEPT ===");
		  for(int i=0; i<m_classifierNC.size();i++)
			  System.out.println(sdf.format(calendar.getTime())+"\t Classifier "+(i+1)+" : "+m_classifierNC.get(i).affich_nom_rule(true)+"\n");
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
  
  public ArrayList<Classification_Rule> ExtraireRegleFermNom(Instances inst, int critere) throws Exception 
  {
	  
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

	    
		/* Génération d'un classifieur de type CNC à partir de la fermeture 
	     * de la valeur nominale la plus pertienente (qui maximise le Support) 
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
				
					//Exctration des indices d'instances etiquitées par la classe d'indice <cl>
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
	       
	    return classifierNC;
	    
}
      

  public ArrayList<Classification_Rule> ExtraireRegleFermAllNom(Instances inst, int critere) throws Exception {
	  
		ArrayList <Classification_Rule> classifierAllNC= new ArrayList<Classification_Rule>(); 
				
		if(m_Debug)
		{
			System.out.println("\nAffichage du context non binaire");
			System.out.println("\tListe des attributs:");
			System.out.print("\t");
			for(int i=0; i<inst.numAttributes();i++)
				System.out.print("("+(i+1)+") "+inst.attribute(i).name()+"  ");
			System.out.println("\n\tContext:");
			for (int i=0 ; i<inst.numInstances(); i++)
				System.out.println("\t"+(i+1)+" : "+inst.instance(i).toString());
		}
		
		for (int ind_attr=0 ; ind_attr<inst.numAttributes()-1 ; ind_attr++)
		{	    
			Attribute m_Attribute;
			m_Attribute = inst.attribute(ind_attr);
			if(m_Debug)
			{	
		    	System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_Attribute.name());
		    	System.out.println("\tAttribut d'indice : "+m_Attribute.index());
		    	System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_Attribute.index()));
		    	for(int i=0; i<inst.numDistinctValues(m_Attribute.index()); i++)
		    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_Attribute.index()).value(i));
			}
	    
			/* Génération d'un classifieur de type CNC à partir de la fermeture 
		     * de la valeur nominale la plus pertienente (qui maximise le Support) 
		     */
		    if(critere == 1) 
		    {
		    	if(m_Debug)
		    		System.out.println("\nGénération d'un CNC à partir de la fermeture du valeur la plus pertinente de l'attribut selectionné");
				
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
					//System.out.println("Support du DistinctValue '"+inst.attribute(m_Attribute.index()).value(i)+"' : "+supportDistVal);
					if(suppBestDistVal <= supportDistVal)
					{
						suppBestDistVal=supportDistVal;
						indexBestDistVal = i;
					}
				}	
				
				if(m_Debug)
					System.out.println("Meilleur DistinctValue: '"+m_Attribute.value(indexBestDistVal)+"' avec un support qui vaut: "+suppBestDistVal);
				
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
				
				Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax),-1);
				//System.out.println (r.affich_nom_rule(true));
				
				// Pondération du classifier (régle) en cours 
				double tmp_pond=0;
				Classify_Instance  tmp_listRules = new Classify_Instance();
				for(int i=0; i<inst.numInstances();i++)
				{
					//System.out.println (inst.get(i).toString());
					if (tmp_listRules.classify_Instance_nom(inst.get(i), (Classification_Rule) r) != -1.0)
						tmp_pond++;  
					//System.out.println (tmp_pond);
				}
				r.setRule_Ponderation( (double) tmp_pond / inst.size() ); 
				
				classifierAllNC.add(r);	
		    }
	    
		    /* Génération d'un classifieur de type CNC à partir de la fermeture 
		     * de chaque valeur nominale de l'attribut nominal selectionné
		     */
		    if(critere == 2)
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
						
						if(m_Debug)	
						{
							System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
							System.out.println ("Liste des attributs de la fermeture");
							for (int o=0;o<FERM_att.size();o++)
								System.out.print (FERM_att.get(o)+" , ");
							System.out.println ("");
						}
								
						//On retourne le concept Pertinent comme un vecteur 
						ArrayList <String> CP= new ArrayList <String>();
						for (int i=0; i< (int) inst.numAttributes()-1;i++)
								CP.add(FERM_att.get(i));	        
						
						Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax),-1);
						
						// Pondération du classifier (régle) en cours 
						double tmp_pond=0;
						Classify_Instance  tmp_listRules = new Classify_Instance();
						for(int i=0; i<inst.numInstances();i++)
						{
							//System.out.println (inst.get(i).toString());
							if (tmp_listRules.classify_Instance_nom(inst.get(i), (Classification_Rule) r) != -1.0)
								tmp_pond++;  
							//System.out.println (tmp_pond);
						}
						r.setRule_Ponderation( (double)tmp_pond / inst.size() ); 
						
						classifierAllNC.add(r);

					} // end if(instDistVal.size()!=0)
				} // end for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_Attribute.index()); indDistVal++)
		    } // end if(critere == 2)
	    
	    } // end for (int ind_attr=0 ; ind_attr<inst.numAttributes()-1 ; ind_attr++)
		
		if(m_Debug)	
		for (int i=0 ; i< classifierAllNC.size() ; i++ )
			System.out.println (classifierAllNC.get(i).affich_nom_rule(true));
	       
	    return classifierAllNC;
	    
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

     
	 	 
  public String toString() 
  {
	Calendar calendar = Calendar.getInstance(); 
	SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS"); 
	//System.err.println("\n"+sdf.format(calendar.getTime())+ "\t");
		
	StringBuffer sb = new StringBuffer(sdf.format(calendar.getTime())+ "\t [Classifier all Nominal Concept Notes] \n"); 
	calendar = Calendar.getInstance();
	
	if((m_CNC==1)&&(this.m_ZeroR==null))
	{	
		sb.append(sdf.format(calendar.getTime())+ "\t Running OK ... \n"); 
		sb.append(sdf.format(calendar.getTime())+ "\t The next display can be missing ... \n"); 
		for(int i=0; i<m_classifierNC.size();i++)
			sb.append(sdf.format(calendar.getTime())+"\t Classifier "+(i+1)+" : "+m_classifierNC.get(i).affich_nom_rule(true)+"\n");
		if(m_Debug)
			sb.append(sdf.format(calendar.getTime())+" \t A Log Console is generated... \n");
		return sb.toString();
	}
	else
		return sdf.format(calendar.getTime())+" \t m_ZeroR model \n" ;
}
  
  
  


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
  runClassifier(new CANC(), argv);
}

@Override
public void updateClassifier(Instance instance) throws Exception {
	// TODO Auto-generated method stub
	System.out.println("/** updateClassifier(Instance instance) **/");  
	
}


































public void DataSetSpecification(Instances data) throws IOException 
{	
	  
//	  ArrayList <Integer> DifferentInst = new ArrayList<Integer>();
	  ArrayList <Integer> TabInstConflict = new ArrayList<Integer>();
	  ArrayList <Integer> InstNotConflict = new ArrayList<Integer>();
//	  ArrayList <Integer> InstWithoutPerturbation = new ArrayList<Integer>();

//	  for (int i=0; i<data.numInstances(); i++)
//	  {
//		  DifferentInst.add(i);
//		  InstNotConflict.add(i);
//		  InstWithoutPerturbation.add(i);
//	  }

		  GregorianCalendar calend=new GregorianCalendar();
		  String DateFile = new String();
		  DateFile = ""
				  +calend.get(GregorianCalendar.YEAR)+"."
				  +calend.get(GregorianCalendar.MONTH)+"."
				  +calend.get(GregorianCalendar.DAY_OF_MONTH)+"-"
				  +calend.get(GregorianCalendar.HOUR_OF_DAY)+"."
				  +calend.get(GregorianCalendar.MINUTE)+"."
				  +calend.get(GregorianCalendar.SECOND)+"."
				  +calend.get(GregorianCalendar.MILLISECOND);
		  
		  //Création d'un fichier contenant des informations à propos de la data set
		  //PrintWriter ecrivainClassifieurs =  new PrintWriter(new BufferedWriter (new FileWriter(System.getProperty("user.home")+"\\"+inst.relationName()+".dss")));
		  //PrintWriter ecrivainClassifieurs =  new PrintWriter(new BufferedWriter (new FileWriter(System.getProperty("user.home")+"\\"+"BNC-"+MatriculeFile+".dss")));
		  
		  File SpecificationDS=new File ("C:\\Data Sets Specification"); 
		  SpecificationDS.mkdirs();
		  
		  String MatriculeFile = new String();
		  MatriculeFile = DateFile + "-" + data.relationName();
		  PrintWriter ecrivainSpecificationDS =  new PrintWriter(new BufferedWriter (new FileWriter("C:\\Data Sets Specification\\DSS-"
				  +MatriculeFile+".dss")));
		  ecrivainSpecificationDS.println("SYSTEM:             DATA SET SPECIFICATION");
		  ecrivainSpecificationDS.println("DATE :              "+DateFile);
		  ecrivainSpecificationDS.println("PATH:               "+"C:\\Data Sets Specification\\DSS-"+MatriculeFile+".dss");
		  ecrivainSpecificationDS.println("CONTEXT:            "+data.relationName());
		  ecrivainSpecificationDS.println("CARACTERITCS LISTE: "
				  +"\n \t\t\t\t INSTANCES:                             "+ data.numInstances()
				  +"\n \t\t\t\t ATRIBUTES (including class attribut):  "+ data.numAttributes()
				  +"\n \t\t\t\t CLASS:                                 "+ data.numClasses()
				  +"\n");
			
		  // System.err.println(data.toString());
		  
		  // initialiser la repartition des attributs...
		  ArrayList <ArrayList <Integer>> RepartAtt = new ArrayList <ArrayList <Integer>>(); 
		  for(int i=0; i<data.numAttributes();i++)
		  {
			  ArrayList <Integer> tempRepartAtt = new ArrayList <Integer>();
			  for(int j=0; j<data.attribute(i).numValues();j++)
				  tempRepartAtt.add(0);
			  RepartAtt.add(tempRepartAtt);
		  }
		  
		  //Lister les attributs
		  ecrivainSpecificationDS.println("\nListe des attributs: ");
		  for(int i=0; i<data.numAttributes();i++)
			  ecrivainSpecificationDS.println("\t"+data.attribute(i));
		  
		  for(int i=0; i<data.numInstances();i++)
			  for(int j=0; j<data.numAttributes();j++)
				  for(int k=0; k<data.attribute(j).numValues();k++)
					  if(data.instance(i).stringValue(j)==data.attribute(j).value(k))
					  {
						  int increm = RepartAtt.get(j).get(k)+1;
						  RepartAtt.get(j).set(k,increm);
					  }
		  
		  ecrivainSpecificationDS.println("\nRépartition des valeurs de chaque attribut: ");
		  for(int i=0; i<data.numAttributes();i++)
		  {
			  ecrivainSpecificationDS.println("Pour l'attribut: '"+data.attribute(i).name()+"' : ");
			  for(int j=0; j<data.attribute(i).numValues();j++)
				  ecrivainSpecificationDS.println("\t\t* '"+data.attribute(i).value(j)+"' : "
						  +RepartAtt.get(i).get(j)+" ( "+(RepartAtt.get(i).get(j)*100.0/data.numInstances())+"% AllInst )");
			  }
			
		  //Lister les valeurs de l'attribut caractérisant la classe
		  ecrivainSpecificationDS.println("\nListe des valeurs de l'attribut classe '"+data.attribute(data.classIndex()).name()+"' ");
		  for(int i=0; i<data.numClasses();i++)
			  ecrivainSpecificationDS.println("\t"+data.attribute(data.classIndex()).value(i));

		  // Extraction d'un contexte non étiquité  
		  Instances dataMissingClass = new Instances (data);
		  for (int i=0; i<dataMissingClass.numInstances();i++)
			  dataMissingClass.instance(i).setClassMissing();
//			  dataMissingClass.instance(i).deleteAttributeAt();
//		  System.out.println(dataMissingClass.toString());

		  // Répartition des classes
		  ArrayList <ArrayList <Integer>> RepartClass = new ArrayList <ArrayList <Integer>>(); 
		  for(int k=0; k<data.numDistinctValues(data.classAttribute());k++)
		  {
			  ArrayList <Integer> tempRepartClass = new ArrayList <Integer>();
			  RepartClass.add(tempRepartClass);
		  }
		  for(int i=0; i<data.numInstances();i++)
			  for(int j=0; j<data.numDistinctValues(data.classAttribute());j++)
				  if(data.instance(i).stringValue(data.classIndex()) == data.attribute(data.classIndex()).value(j))
					  RepartClass.get(j).add(i);
		  
		  ecrivainSpecificationDS.println("\nListe des indices d'instances par classe :");
		  for(int i=0; i<data.numClasses();i++)
			  ecrivainSpecificationDS.println("\t"+(i+1)+" th class ( "+data.attribute(data.classIndex()).value(i)+" ) : "
					  +RepartClass.get(i).size()+" instances ( "+(RepartClass.get(i).size()*100.2/data.numInstances())
					  +"% AllInst ): "+RepartClass.get(i));

		  //lister les ensembles des instances dupliquées 
		  ecrivainSpecificationDS.print("\nDetection des ensembles d'instances dupliquées (A partir d'un context étiquité):");
		  
		  ArrayList <ArrayList <Integer>> DupInst = new ArrayList <ArrayList <Integer>>();
		  ArrayList <Integer> NotDupInst = new ArrayList <Integer>();
		  
		  ArrayList <ArrayList <Integer>> RepartClassDupInst = new ArrayList <ArrayList <Integer>>();
		  for(int k=0; k<data.numDistinctValues(data.classAttribute());k++)
		  {
			  ArrayList <Integer> tempRepartClassDupInst = new ArrayList <Integer>();
			  RepartClassDupInst.add(tempRepartClassDupInst);
		  }
		  
		  
		  ArrayList <Integer> SommeRepartClassDupInst = new ArrayList <Integer>();
		  for(int k=0; k<data.numDistinctValues(data.classAttribute());k++)
		  {
			  SommeRepartClassDupInst.add(0);
		  }
		  
		  int maxEns=0;
		  
		  for(int i=0; i<data.numInstances();i++)
			{
			  ArrayList <Integer> UnEnsDupInst = new ArrayList <Integer>();
				UnEnsDupInst.clear();			
				UnEnsDupInst.add(i); //Commencer par inserer l'indice de la i émé à la recherche des redondants
				
				for (int j=i+1; j<(int)data.numInstances();j++)
				{ 
					if(data.instance(i).toString().equals( data.instance(j).toString()) && InstNotIn(j,DupInst)) 
					{ 
						UnEnsDupInst.add(j);  
					}
				}

				if (UnEnsDupInst.size()>1)
				{
					DupInst.add(UnEnsDupInst);
					if(maxEns<UnEnsDupInst.size())
						maxEns=UnEnsDupInst.size();
				}
			}
		  
			int SommeDup=0;
			if(DupInst.isEmpty())
			{
				ecrivainSpecificationDS.println("\n\t\tAucun ensemble d'instances dupliquées..." +
						"\n\t\tLes données de ce context sont divers à 100%");
				ecrivainSpecificationDS.println("\t\tNombre des instances différentes: "+ data.numInstances());
			}
			else
			{			
				for(int i=0; i<data.numInstances();i++)
					if (this.InstNotIn(i, DupInst))
						NotDupInst.add(i);
				
				ecrivainSpecificationDS.print("\n\nTrier les ensembles d'instances dupliquées:");
				for(int i=0; i<NotDupInst.size();i++)
					ecrivainSpecificationDS.print("\n\t"+(i+1)+" th Ensemble (1 instances, "
							+(100.0/data.numInstances())+"% AllInst ): ["+ NotDupInst.get(i) +"]");
				
				int delimiteur=2;
				int tempCpt = NotDupInst.size();
				while(delimiteur<=maxEns)
				{
					for(int i=0; i<DupInst.size(); i++)
					{
						if(delimiteur == DupInst.get(i).size())
						{
							ecrivainSpecificationDS.print("\n\t"+(tempCpt+1)+" th Ensemble ("
									+DupInst.get(i).size()+" instances, "
									+(DupInst.get(i).size()*100.0/data.numInstances())+"% AllInst): "
									+DupInst.get(i));
							tempCpt++;
							SommeDup+=DupInst.get(i).size();
						}
					}
					delimiteur ++;
							
				}
					
				
				for(int i=0; i<(int)DupInst.size();i++)
					  for(int k=0; k<data.numDistinctValues(data.classAttribute());k++)
						  if(data.instance(DupInst.get(i).get(0)).stringValue(data.classIndex()) 
								  == data.attribute(data.classIndex()).value(k))
						  {
							  RepartClassDupInst.get(k).add(i+1);
							  SommeRepartClassDupInst.set(k, SommeRepartClassDupInst.get(k)+DupInst.get(i).size());						  
						  }
				  
				  ecrivainSpecificationDS.println("\n\nListe des ensembles d'instances (indice) dupliquées par classe :");
				  for(int i=0; i<data.numClasses();i++)
				  {							  
					  ecrivainSpecificationDS.println("\t"+(i+1)+" th class ( "+data.attribute(data.classIndex()).value(i)+" ) : ( "
							  +(SommeRepartClassDupInst.get(i)*100.0/SommeDup)+"% AllDup - "
							  +(SommeRepartClassDupInst.get(i)*100.0/data.numInstances())+"% AllInst ) "
							  +RepartClassDupInst.get(i).size()+" ensembles : "+RepartClassDupInst.get(i));
				  }
			  ecrivainSpecificationDS.println("\n\t\tNombre des instances dupliquées: "+ SommeDup
						+" ( "+(SommeDup*100.0/data.numInstances())+"% AllInst )");
				ecrivainSpecificationDS.println("\t\tNombre des instances non dupliquées: "+ NotDupInst.size()
						+" ( "+(NotDupInst.size()*100.0/data.numInstances())+"% AllInst )");
				ecrivainSpecificationDS.println("\t\tNombre des instances différentes: "+ (DupInst.size()+NotDupInst.size())
						+" ( "+((DupInst.size()+NotDupInst.size())*100.0/data.numInstances())+"% AllInst )");			  
			}
			

		/*
		 * Parcours des instances à la Recherches des instances en conflit de classe (y compris les dupliquees)
		 */

		ecrivainSpecificationDS.print("\nDetection des instances en conflit de classe (y compris les dupliquees):");
		ArrayList <ArrayList <Integer>> CfDpInst = new ArrayList <ArrayList <Integer>>();
		
		for (int i=0; i<(int)data.numInstances();i++)
		{
			ArrayList <Integer> UnEnsCfDpInst = new ArrayList <Integer>();
			UnEnsCfDpInst.clear();
			UnEnsCfDpInst.add(i);

			if (InstNotIn(i+1,CfDpInst)== true )
			{
				for (int j=i+1; j<(int)data.numInstances();j++)
				{ 					
					if(dataMissingClass.instance(i).toString().equals(dataMissingClass.instance(j).toString()) && InstNotIn(j,DupInst)) 
					{ 
						UnEnsCfDpInst.add(j); 
					}
				}
			}
			if (UnEnsCfDpInst.size()>1)
			{
				CfDpInst.add(UnEnsCfDpInst);
				for(int z=0; z<UnEnsCfDpInst.size();z++)
					if(this.InstNotInS(UnEnsCfDpInst.get(z), TabInstConflict))
						TabInstConflict.add(UnEnsCfDpInst.get(z));
			}
		}
		
		int SommeCfDp=0;
		if(CfDpInst.isEmpty())
			ecrivainSpecificationDS.println("\n\t\tAucun ensemble d'instances en conflit de classe (y compris les dupliquees) ");
		else
		{
			this.trier_tableau(TabInstConflict);
			ecrivainSpecificationDS.println("\nListe des instances en conflit de classe (y compris les dupliquees): "+TabInstConflict.toString());

			Instances dataWithoutConflict = new Instances(data);
			for(int i=TabInstConflict.size()-1; i>=0 ;i--)
				dataWithoutConflict.delete(TabInstConflict.get(i));
			CreateDSwithoutConflict(dataWithoutConflict);
				
			for (int i=0; i<(int)CfDpInst.size();i++)
			{
				ecrivainSpecificationDS.print("\n\t"+(i+1)+" th Ensemble d'indice: "+CfDpInst.get(i));
//				for(int j=0; j<(int) CfDpInst.get(i).size(); j++)
//					ecrivainSpecificationDS.print("  "+CfDpInst.get(i).get(j));	
				SommeCfDp+=CfDpInst.get(i).size();
			}
		}
		ecrivainSpecificationDS.println("\n\t\tNombre des instances en conflit de classe (y compris les dupliquees): "+ SommeCfDp
//				+"(" +(SommeCfDp*100./data.numInstances())+"% AllInst)"
				);

		/*
		 * Parcours des instances à la Recherches des instances en conflit de classe (sans les dupliquees)
		 */
		
		ecrivainSpecificationDS.print("\nDetection des instances en conflit de classe (sans les dupliquees) (Fisrt):");
		ArrayList <ArrayList <Integer>> FirstCfInst = new ArrayList <ArrayList <Integer>>();
		
		for (int i=0; i<(int)data.numInstances();i++)
		{
			ArrayList <Integer> FirstUnEnsCfInst = new ArrayList <Integer>();
			FirstUnEnsCfInst.clear();
			FirstUnEnsCfInst.add(i);
		
			if (InstNotIn(i+1,FirstCfInst)== true 
					&& InstNotIn(i+1,DupInst)== true
					)
			{
//				Instance instanceI = new Instance(data.instance(i));
//				instanceI.deleteAttributeAt(data.classIndex());
				for (int j=i+1; j<(int)data.numInstances();j++)
				{ 
//					Instance instanceJ = new Instance(data.instance(j));
//					instanceJ.deleteAttributeAt(data.classIndex());
					
//					if(instanceI.toString().equals(instanceJ.toString()) 
					if(dataMissingClass.instance(i).toString().equals(dataMissingClass.instance(j).toString())
							 && InstNotIn(j,DupInst)) 
					{ 
						FirstUnEnsCfInst.add(j); 
					}
				}
			}
			if (FirstUnEnsCfInst.size()>1)
			{
				FirstCfInst.add(FirstUnEnsCfInst);
			}
		}
		
		int FirstSommeCf=0;
		if(FirstCfInst.isEmpty())
			ecrivainSpecificationDS.println("\n\t\tAucun ensemble d'instances en conflit de classe (sans les dupliquees)(First)");
		else
			for (int i=0; i<(int)FirstCfInst.size();i++)
			{
				ecrivainSpecificationDS.print("\n\t"+(i+1)+" th Ensemble d'indice: "+FirstCfInst.get(i));
//				for(int j=0; j<(int) FirstCfInst.get(i).size(); j++)
//					ecrivainSpecificationDS.print("  "+FirstCfInst.get(i).get(j));	
				FirstSommeCf+=FirstCfInst.get(i).size();
			}
		ecrivainSpecificationDS.println("\n\t\tNombre des instances en conflit de classe (sans les dupliquees) (First): "+ FirstSommeCf
//				+"(" +(FirstSommeCf*100./data.numInstances())+"%)"
				);

		/*
		 * Parcours des instances à la Recherches des instances en conflit de classe (sans les dupliquees)
		 */
		
		ecrivainSpecificationDS.print("\nDetection des instances en conflit de classe (sans les dupliquees)(All):");
		ArrayList <ArrayList <Integer>> AllCfInst = new ArrayList <ArrayList <Integer>>();
		
		for (int i=0; i<(int)data.numInstances();i++)
		{
			ArrayList <Integer> AllUnEnsCfInst = new ArrayList <Integer>();
			AllUnEnsCfInst.clear();
			AllUnEnsCfInst.add(i);
		
			if (InstNotIn(i+1,AllCfInst)== true 
//					&& InstNotIn(i+1,DupInst)== true
					)
			{
//				Instance instanceI = new Instance(data.instance(i));
//				instanceI.deleteAttributeAt(data.classIndex());
				for (int j=i+1; j<(int)data.numInstances();j++)
				{ 
//					Instance instanceJ = new Instance(data.instance(j));
//					instanceJ.deleteAttributeAt(data.classIndex());
					
//					if(instanceI.toString().equals(instanceJ.toString()) 
					if(dataMissingClass.instance(i).toString().equals(dataMissingClass.instance(j).toString())
							&& !data.instance(i).toString().equals( data.instance(j).toString())
							 && InstNotIn(j,DupInst)
							) 
					{ 
						AllUnEnsCfInst.add(j); 
					}
				}
			}
			if (AllUnEnsCfInst.size()>1)
			{
				AllCfInst.add(AllUnEnsCfInst);
			}
		}
		
		int AllSommeCf=0;
		if(AllCfInst.isEmpty())
			ecrivainSpecificationDS.println("\n\t\tAucun ensemble d'instances en conflit de classe (sans les dupliquees)(All).");
		else
			for (int i=0; i<(int)AllCfInst.size();i++)
			{
				ecrivainSpecificationDS.print("\n\t"+(i+1)+" th Ensemble d'indice: "+AllCfInst.get(i));
//				for(int j=0; j<(int) AllCfInst.get(i).size(); j++)
//					ecrivainSpecificationDS.print("  "+AllCfInst.get(i).get(j));	
				AllSommeCf+=AllCfInst.get(i).size();
			}
		ecrivainSpecificationDS.println("\n\t\tNombre des instances en conflit de classe (sans les dupliquees) (All): "+ AllSommeCf
//				+"(" +(AllSommeCf*100./data.numInstances())+"%)"
				);

		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//Parcours des instances à la recherches des missing values
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////
/*		
		ArrayList <ArrayList <Integer>> RepartAttMV = new ArrayList <ArrayList <Integer>>(); 
		  for(int i=0; i<data.numAttributes();i++)
		  {
			  ArrayList <Integer> tempRepartAttMV = new ArrayList <Integer>();
			  RepartAttMV.add(tempRepartAttMV);
		  }
		  // Intilalistaion par des zero
		  for(int i=0; i<data.numAttributes();i++)
			  for(int j=0; j<data.attribute(j).numValues();j++)
				  RepartAttMV.get(i).add((int)0);

		int nbre_mv=0;
		ArrayList <ArrayList <Integer>> InstMV = new ArrayList <ArrayList <Integer>>();
		ArrayList <ArrayList <Integer>> RepartClassInstMV = new ArrayList <ArrayList <Integer>>();
		for(int k=0; k<data.numDistinctValues(data.classAttribute());k++)
		{
			ArrayList <Integer> tempRepartClassInstMV = new ArrayList <Integer>();
			RepartClassInstMV.add(tempRepartClassInstMV);
		}
		ArrayList <ArrayList <Integer>> RepartNBREInstMV = new ArrayList <ArrayList <Integer>>();
		for(int k=0; k<data.numAttributes();k++)
		{
			ArrayList <Integer> tempRepartNBREInstMV = new ArrayList <Integer>();
			RepartNBREInstMV.add(tempRepartNBREInstMV);
		}
		
		ecrivainSpecificationDS.println("\n\nDetection des instances qui contiennent des missing values:");
		for (int i=0; i<data.numInstances();i++)
		{
			if(data.instance(i).hasMissingValue())
			{				
				ArrayList <Integer> UnInstMV = new ArrayList <Integer>();
				UnInstMV.clear();
				UnInstMV.add(i);
				for(int k=0; k<data.instance(i).numAttributes(); k++)
					if(data.instance(i).isMissing(k))
					{
						UnInstMV.add(k);
						 for(int h=0; h<data.attribute(k).numValues();h++)
							 if(data.instance(i).stringValue(k)==data.attribute(k).value(h))
							 {
								 int increm = RepartAttMV.get(k).get(h)+1;
								 RepartAttMV.get(k).set(h,increm);
							 }							
					}
				InstMV.add(UnInstMV);
				nbre_mv+=UnInstMV.size()-1;
				RepartNBREInstMV.get((int)UnInstMV.size()-2).add(i);
				for(int k=0; k<data.numClasses();k++)
					  if(data.instance(i).stringValue(data.classIndex()) 
							  == data.attribute(data.classIndex()).value(k))
						  RepartClassInstMV.get(k).add(i);
			}
		}

		if(nbre_mv==0)
			ecrivainSpecificationDS.println("\n\t\tAucune missing value dans cet echantillon de données");
		else
		{
			for (int i=0; i<(int)InstMV.size();i++)
			{
				ecrivainSpecificationDS.print("\n\t"+(InstMV.get(i).get(0)+1)+" th instance: "+(InstMV.get(i).size()-1)+" MV : ");
				for(int j=1; j<InstMV.get(i).size();j++)
				{
					ecrivainSpecificationDS.print(" -- "+data.attribute((InstMV.get(i).get(j))).name());
				}
			}
			
			ecrivainSpecificationDS.println("\n\nRepartition des indices d'instances contenant des Missing Values selon les classes: ");
			for(int k=0; k<data.numClasses();k++)
				ecrivainSpecificationDS.println("\t"+(k+1)+" th class ("+data.attribute(data.classIndex()).value(k)+") : "
						  +RepartClassInstMV.get(k).size()+" indices d'instances : ( "
						  +(RepartClassInstMV.get(k).size()*100.2/InstMV.size())+"% AllInstMV - "
						  +(RepartClassInstMV.get(k).size()*100.2/data.numInstances())+"% AllInst )"
						  +RepartClassInstMV.get(k));
			
			ecrivainSpecificationDS.println("\n\nRepartition des indices d'instances selon les nombres de Missing Values contenus: ");
			for(int k=0; k<RepartNBREInstMV.size();k++)
				ecrivainSpecificationDS.println("\t"+(k+1)+" th ensemble ( "+(k+1)+ "MV - "
						+(RepartNBREInstMV.get(k).size()*100.2/nbre_mv)+"% AllMV - "
						+(RepartNBREInstMV.get(k).size()*100.2/(data.numAttributes()*data.numInstances()))
						+"% AllValues ) :  indices d'instances : "+RepartNBREInstMV.get(k));
			
			  ecrivainSpecificationDS.println("\nRépartitions des valeurs des attributs via les Missing Values: ");
			  for(int i=0; i<data.numAttributes();i++)
			  {
				  ecrivainSpecificationDS.println("\t"+"Pour l'attribut: '"+data.attribute(i).name()+"' : ");
				  for(int j=0; j<data.attribute(i).numValues();j++)
					  ecrivainSpecificationDS.println("\t\t* '"
//							  +data.attribute(i).value(j)
//							  +"' : "+RepartAttMV.get(i).get(j)
//							  +" ( "+(RepartAttMV.get(i).get(j)*100.2/data.numInstances())+"% AllInst )"
							  )
							  ;
			  }

			
			
			
			ecrivainSpecificationDS.println("\n\t\tNombre des missing values: " 
					+ nbre_mv+" ( "+(nbre_mv*100./(data.numAttributes() * data.numInstances()))+"% AllValues)");
		}
*/
		ecrivainSpecificationDS.close();

}

public void DataSetSpecificationLight(Instances data) throws IOException 
{	
	  ArrayList <Integer> TabInstConflict = new ArrayList<Integer>();
	  ArrayList <Integer> InstNotConflict = new ArrayList<Integer>();

		  GregorianCalendar calend=new GregorianCalendar();
		  String DateFile = new String();
		  DateFile = ""
				  +calend.get(GregorianCalendar.YEAR)+"."
				  +calend.get(GregorianCalendar.MONTH)+"."
				  +calend.get(GregorianCalendar.DAY_OF_MONTH)+"-"
				  +calend.get(GregorianCalendar.HOUR_OF_DAY)+"."
				  +calend.get(GregorianCalendar.MINUTE)+"."
				  +calend.get(GregorianCalendar.SECOND)+"."
				  +calend.get(GregorianCalendar.MILLISECOND);
		  
		  //Création d'un fichier contenant des informations à propos de la data set
		  //PrintWriter ecrivainClassifieurs =  new PrintWriter(new BufferedWriter (new FileWriter(System.getProperty("user.home")+"\\"+inst.relationName()+".dss")));
		  //PrintWriter ecrivainClassifieurs =  new PrintWriter(new BufferedWriter (new FileWriter(System.getProperty("user.home")+"\\"+"BNC-"+MatriculeFile+".dss")));
		  
		  File SpecificationDS=new File ("C:\\Data Sets Specification"); 
		  SpecificationDS.mkdirs();
		  
		  String MatriculeFile = new String();
		  MatriculeFile = DateFile + "-" + data.relationName();
		  PrintWriter ecrivainSpecificationDS =  new PrintWriter(new BufferedWriter (new FileWriter("C:\\Data Sets Specification\\DSS-"
				  +MatriculeFile+".dss")));
		  ecrivainSpecificationDS.println("SYSTEM:             DATA SET SPECIFICATION");
		  ecrivainSpecificationDS.println("DATE :              "+DateFile);
		  ecrivainSpecificationDS.println("PATH:               "+"C:\\Data Sets Specification\\DSS-"+MatriculeFile+".dss");
		  ecrivainSpecificationDS.println("CONTEXT:            "+data.relationName());
		  ecrivainSpecificationDS.println("CARACTERITCS LISTE: "
				  +"\n \t\t\t\t INSTANCES:                             "+ data.numInstances()
				  +"\n \t\t\t\t ATRIBUTES (including class attribut):  "+ data.numAttributes()
				  +"\n \t\t\t\t CLASS:                                 "+ data.numClasses()
				  +"\n");
			
		  // System.err.println(data.toString());
		  
		  // initialiser la repartition des attributs...
		  ArrayList <ArrayList <Integer>> RepartAtt = new ArrayList <ArrayList <Integer>>(); 
		  for(int i=0; i<data.numAttributes();i++)
		  {
			  ArrayList <Integer> tempRepartAtt = new ArrayList <Integer>();
			  for(int j=0; j<data.attribute(i).numValues();j++)
				  tempRepartAtt.add(0);
			  RepartAtt.add(tempRepartAtt);
		  }
		  
		  //Lister les attributs
		  ecrivainSpecificationDS.println("\nListe des attributs: ");
		  for(int i=0; i<data.numAttributes();i++)
			  ecrivainSpecificationDS.println("\t"+data.attribute(i));
		  
		  for(int i=0; i<data.numInstances();i++)
			  for(int j=0; j<data.numAttributes();j++)
				  for(int k=0; k<data.attribute(j).numValues();k++)
					  if(data.instance(i).stringValue(j)==data.attribute(j).value(k))
					  {
						  int increm = RepartAtt.get(j).get(k)+1;
						  RepartAtt.get(j).set(k,increm);
					  }

		  // Extraction d'un contexte non étiquité  
		  Instances dataMissingClass = new Instances (data);
		  for (int i=0; i<dataMissingClass.numInstances();i++)
			  dataMissingClass.instance(i).setClassMissing();
//			  dataMissingClass.instance(i).deleteAttributeAt();
//		  System.out.println(dataMissingClass.toString());

		  // Répartition des classes
		  ArrayList <ArrayList <Integer>> RepartClass = new ArrayList <ArrayList <Integer>>(); 
		  for(int k=0; k<data.numDistinctValues(data.classAttribute());k++)
		  {
			  ArrayList <Integer> tempRepartClass = new ArrayList <Integer>();
			  RepartClass.add(tempRepartClass);
		  }
		  for(int i=0; i<data.numInstances();i++)
			  for(int j=0; j<data.numDistinctValues(data.classAttribute());j++)
				  if(data.instance(i).stringValue(data.classIndex()) == data.attribute(data.classIndex()).value(j))
					  RepartClass.get(j).add(i);
		  
	  
		  ArrayList <ArrayList <Integer>> DupInst = new ArrayList <ArrayList <Integer>>();
		  ArrayList <Integer> NotDupInst = new ArrayList <Integer>();
		  
		  ArrayList <ArrayList <Integer>> RepartClassDupInst = new ArrayList <ArrayList <Integer>>();
		  for(int k=0; k<data.numDistinctValues(data.classAttribute());k++)
		  {
			  ArrayList <Integer> tempRepartClassDupInst = new ArrayList <Integer>();
			  RepartClassDupInst.add(tempRepartClassDupInst);
		  }
		  
		  
		  ArrayList <Integer> SommeRepartClassDupInst = new ArrayList <Integer>();
		  for(int k=0; k<data.numDistinctValues(data.classAttribute());k++)
		  {
			  SommeRepartClassDupInst.add(0);
		  }
		  
		  int maxEns=0;
		  
		  for(int i=0; i<data.numInstances();i++)
			{
			  ArrayList <Integer> UnEnsDupInst = new ArrayList <Integer>();
				UnEnsDupInst.clear();			
				UnEnsDupInst.add(i); //Commencer par inserer l'indice de la i émé à la recherche des redondants
				
				for (int j=i+1; j<(int)data.numInstances();j++)
				{ 
					if(data.instance(i).toString().equals( data.instance(j).toString()) && InstNotIn(j,DupInst)) 
						UnEnsDupInst.add(j);  
				}

				if (UnEnsDupInst.size()>1)
				{
					DupInst.add(UnEnsDupInst);
					if(maxEns<UnEnsDupInst.size())
						maxEns=UnEnsDupInst.size();
				}
			}
		  
			int SommeDup=0;
			if(DupInst.isEmpty())
			{
				ecrivainSpecificationDS.println("\n\t\tAucun ensemble d'instances dupliquées..." +
						"\n\t\tLes données de ce context sont divers à 100%");
				ecrivainSpecificationDS.println("\t\tNombre des instances différentes: "+ data.numInstances());
			}
			else
			{			
				for(int i=0; i<data.numInstances();i++)
					if (this.InstNotIn(i, DupInst))
						NotDupInst.add(i);
				
				ecrivainSpecificationDS.print("\n\nTrier les ensembles d'instances dupliquées:");
//				for(int i=0; i<NotDupInst.size();i++)
//					ecrivainSpecificationDS.print("\n\t"+(i+1)+" th Ensemble (1 instances, "
//							+(100.2/data.numInstances())+"% AllInst ): ["+ NotDupInst.get(i) +"]");
				
				int delimiteur=2;
				int tempCpt = NotDupInst.size();
				while(delimiteur<=maxEns)
				{
					for(int i=0; i<DupInst.size(); i++)
					{
						if(delimiteur == DupInst.get(i).size())
						{
							ecrivainSpecificationDS.print("\n\t"+(tempCpt+1)+" th Ensemble ("
									+DupInst.get(i).size()+" instances, "
									+(DupInst.get(i).size()*100.0/data.numInstances())+"% AllInst): "
									+DupInst.get(i));
							tempCpt++;
							SommeDup+=DupInst.get(i).size();
						}
					}
					delimiteur ++;
							
				}
					
				
				for(int i=0; i<(int)DupInst.size();i++)
					  for(int k=0; k<data.numDistinctValues(data.classAttribute());k++)
						  if(data.instance(DupInst.get(i).get(0)).stringValue(data.classIndex()) 
								  == data.attribute(data.classIndex()).value(k))
						  {
							  RepartClassDupInst.get(k).add(i+1);
							  SommeRepartClassDupInst.set(k, SommeRepartClassDupInst.get(k)+DupInst.get(i).size());						  
						  }
				  
				  ecrivainSpecificationDS.println("\n\nListe des ensembles d'instances (indice) dupliquées par classe :");
				  for(int i=0; i<data.numClasses();i++)
				  {							  
					  ecrivainSpecificationDS.println("\t"+(i+1)+" th class ( "+data.attribute(data.classIndex()).value(i)+" ) : ( "
							  +(SommeRepartClassDupInst.get(i)*100.0/SommeDup)+"% AllDup - "
							  +(SommeRepartClassDupInst.get(i)*100.0/data.numInstances())+"% AllInst ) "
							  +RepartClassDupInst.get(i).size()+" ensembles : "+RepartClassDupInst.get(i));
				  }
			  ecrivainSpecificationDS.println("\n\t\tNombre des instances dupliquées: "+ SommeDup
						+" ( "+(SommeDup*100.0/data.numInstances())+"% AllInst )");
				ecrivainSpecificationDS.println("\t\tNombre des instances non dupliquées: "+ NotDupInst.size()
						+" ( "+(NotDupInst.size()*100.0/data.numInstances())+"% AllInst )");
				ecrivainSpecificationDS.println("\t\tNombre des instances différentes: "+ (DupInst.size()+NotDupInst.size())
						+" ( "+((DupInst.size()+NotDupInst.size())*100.0/data.numInstances())+"% AllInst )");			  
			}
			

		/*
		 * Parcours des instances à la Recherches des instances en conflit de classe (y compris les dupliquees)
		 */

		ecrivainSpecificationDS.print("\nDetection des instances en conflit de classe (y compris les dupliquees):");
		ArrayList <ArrayList <Integer>> CfDpInst = new ArrayList <ArrayList <Integer>>();
		
		for (int i=0; i<(int)data.numInstances();i++)
		{
			ArrayList <Integer> UnEnsCfDpInst = new ArrayList <Integer>();
			UnEnsCfDpInst.clear();
			UnEnsCfDpInst.add(i);

			if (InstNotIn(i+1,CfDpInst)== true )
			{
				for (int j=i+1; j<(int)data.numInstances();j++)
				{ 					
					if(dataMissingClass.instance(i).toString().equals(dataMissingClass.instance(j).toString()) && InstNotIn(j,DupInst)) 
					{ 
						UnEnsCfDpInst.add(j); 
					}
				}
			}
			if (UnEnsCfDpInst.size()>1)
			{
				CfDpInst.add(UnEnsCfDpInst);
				for(int z=0; z<UnEnsCfDpInst.size();z++)
					if(this.InstNotInS(UnEnsCfDpInst.get(z), TabInstConflict))
						TabInstConflict.add(UnEnsCfDpInst.get(z));
			}
		}
		
		int SommeCfDp=0;
		if(CfDpInst.isEmpty())
			ecrivainSpecificationDS.println("\n\t\tAucun ensemble d'instances en conflit de classe (y compris les dupliquees) ");
		else
		{
			this.trier_tableau(TabInstConflict);
			ecrivainSpecificationDS.println("\nListe des instances en conflit de classe (y compris les dupliquees): "+TabInstConflict.toString());

			Instances dataWithoutConflict = new Instances(data);
			for(int i=TabInstConflict.size()-1; i>=0 ;i--)
				dataWithoutConflict.delete(TabInstConflict.get(i));
			CreateDSwithoutConflict(dataWithoutConflict);
				
			for (int i=0; i<(int)CfDpInst.size();i++)
			{
				ecrivainSpecificationDS.print("\n\t"+(i+1)+" th Ensemble d'indice: "+CfDpInst.get(i));
//				for(int j=0; j<(int) CfDpInst.get(i).size(); j++)
//					ecrivainSpecificationDS.print("  "+CfDpInst.get(i).get(j));	
				SommeCfDp+=CfDpInst.get(i).size();
			}
		}
		ecrivainSpecificationDS.println("\n\t\tNombre des instances en conflit de classe (y compris les dupliquees): "+ SommeCfDp
//				+"(" +(SommeCfDp*100./data.numInstances())+"% AllInst)"
				);

		/*
		 * Parcours des instances à la Recherches des instances en conflit de classe (sans les dupliquees)
		 */
		
		ecrivainSpecificationDS.print("\nDetection des instances en conflit de classe (sans les dupliquees) (Fisrt):");
		ArrayList <ArrayList <Integer>> FirstCfInst = new ArrayList <ArrayList <Integer>>();
		
		for (int i=0; i<(int)data.numInstances();i++)
		{
			ArrayList <Integer> FirstUnEnsCfInst = new ArrayList <Integer>();
			FirstUnEnsCfInst.clear();
			FirstUnEnsCfInst.add(i);
		
			if (InstNotIn(i+1,FirstCfInst)== true 
					&& InstNotIn(i+1,DupInst)== true
					)
			{
//				Instance instanceI = new Instance(data.instance(i));
//				instanceI.deleteAttributeAt(data.classIndex());
				for (int j=i+1; j<(int)data.numInstances();j++)
				{ 
//					Instance instanceJ = new Instance(data.instance(j));
//					instanceJ.deleteAttributeAt(data.classIndex());
					
//					if(instanceI.toString().equals(instanceJ.toString()) 
					if(dataMissingClass.instance(i).toString().equals(dataMissingClass.instance(j).toString())
							 && InstNotIn(j,DupInst)) 
					{ 
						FirstUnEnsCfInst.add(j); 
					}
				}
			}
			if (FirstUnEnsCfInst.size()>1)
			{
				FirstCfInst.add(FirstUnEnsCfInst);
			}
		}
		
		int FirstSommeCf=0;
		if(FirstCfInst.isEmpty())
			ecrivainSpecificationDS.println("\n\t\tAucun ensemble d'instances en conflit de classe (sans les dupliquees)(First)");
		else
			for (int i=0; i<(int)FirstCfInst.size();i++)
			{
				ecrivainSpecificationDS.print("\n\t"+(i+1)+" th Ensemble d'indice: "+FirstCfInst.get(i));
//				for(int j=0; j<(int) FirstCfInst.get(i).size(); j++)
//					ecrivainSpecificationDS.print("  "+FirstCfInst.get(i).get(j));	
				FirstSommeCf+=FirstCfInst.get(i).size();
			}
		ecrivainSpecificationDS.println("\n\t\tNombre des instances en conflit de classe (sans les dupliquees)(Fisrt): "+ FirstSommeCf
//				+"(" +(FirstSommeCf*100./data.numInstances())+"%)"
				);

		/*
		 * Parcours des instances à la Recherches des instances en conflit de classe (sans les dupliquees)
		 */
		
		ecrivainSpecificationDS.print("\nDetection des instances en conflit de classe (sans les dupliquees)(All):");
		ArrayList <ArrayList <Integer>> AllCfInst = new ArrayList <ArrayList <Integer>>();
		
		for (int i=0; i<(int)data.numInstances();i++)
		{
			ArrayList <Integer> AllUnEnsCfInst = new ArrayList <Integer>();
			AllUnEnsCfInst.clear();
			AllUnEnsCfInst.add(i);
		
			if (InstNotIn(i+1,AllCfInst)== true 
//					&& InstNotIn(i+1,DupInst)== true
					)
			{
//				Instance instanceI = new Instance(data.instance(i));
//				instanceI.deleteAttributeAt(data.classIndex());
				for (int j=i+1; j<(int)data.numInstances();j++)
				{ 
//					Instance instanceJ = new Instance(data.instance(j));
//					instanceJ.deleteAttributeAt(data.classIndex());
					
//					if(instanceI.toString().equals(instanceJ.toString()) 
					if(dataMissingClass.instance(i).toString().equals(dataMissingClass.instance(j).toString())
							&& !data.instance(i).toString().equals( data.instance(j).toString())
							 && InstNotIn(j,DupInst)
							) 
					{ 
						AllUnEnsCfInst.add(j); 
					}
				}
			}
			if (AllUnEnsCfInst.size()>1)
			{
				AllCfInst.add(AllUnEnsCfInst);
			}
		}
		
		int AllSommeCf=0;
		if(AllCfInst.isEmpty())
			ecrivainSpecificationDS.println("\n\t\tAucun ensemble d'instances en conflit de classe (sans les dupliquees)(All).");
		else
			for (int i=0; i<(int)AllCfInst.size();i++)
			{
				ecrivainSpecificationDS.print("\n\t"+(i+1)+" th Ensemble d'indice: "+AllCfInst.get(i));
//				for(int j=0; j<(int) AllCfInst.get(i).size(); j++)
//					ecrivainSpecificationDS.print("  "+AllCfInst.get(i).get(j));	
				AllSommeCf+=AllCfInst.get(i).size();
			}
		ecrivainSpecificationDS.println("\n\t\tNombre des instances en conflit de classe (sans les dupliquees)(All): "+ AllSommeCf
//				+"(" +(AllSommeCf*100./data.numInstances())+"%)"
				);

		ecrivainSpecificationDS.close();

}

boolean InstNotIn(int num, ArrayList<ArrayList<Integer>> VectInst)
{
	for(int i=0 ; i < VectInst.size() ; i++ )
		for(int j=0 ; j < (int)VectInst.get(i).size() ; j++ )
			if(num == VectInst.get(i).get(j))
				return false;
	return true;
}

boolean InstNotInS(int val, ArrayList<Integer> VectInst)
{
	for(int i=0 ; i < VectInst.size() ; i++ )
		if(val == VectInst.get(i))
			return false;
	return true;
}

//Trier un tableau (vecteur des entiers)
void trier_tableau(ArrayList <Integer> tab)
{
	int min,index;
	for(int i=0; i<tab.size(); i++)
	{
		min = tab.get(i);
		index = i;
		for(int j=i+1 ; j<tab.size() ; j++)
			if(tab.get(j)<min)
			{
				index = j;
				min = tab.get(j);
			}
		if(index != i)
		{
			tab.set(index, tab.get(i));
			tab.set(i,min);
		}
	}				
}


public void CreateDSwithoutConflict(Instances Data) throws IOException
{
	Data.setRelationName(Data.relationName()+".WithoutConflict");	// Renommer le contexte
	String path = "\\"+Data.relationName()+".arff";	// Définition d'un chemin pour le fichier contenant ce nouveau context
	PrintWriter ecrivainoutPutsClassifieurs =  new PrintWriter(new BufferedWriter(new FileWriter(path))); // Création d'un fichier ARFF contenant ce contexte	
	ecrivainoutPutsClassifieurs.print(Data.toString());
	ecrivainoutPutsClassifieurs.close();
}



}