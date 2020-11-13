package weka.classifiers.fca;

import weka.core.*;

import java.util.ArrayList;
import java.util.Calendar;
import java.lang.*;
import java.text.SimpleDateFormat;

/**
 * @author Meddouri Nida (nida.meddouri@gmail.com)
 * revision: 20160822 
 */

public class Classify_Instance 
{
	
	/**
	 * Un constructeur
	 */
	public Classify_Instance() throws Exception{

	}
	
	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * @param inst : L'instance à prédire ca classe
	 * @param regle: Une régle de classification dont les attributs sont nominaux
	 * @return L'indice de la classe attribuée pour l'instance inst
	 */
	public double classify_Instance_nom(Instance inst, Classification_Rule regle) throws Exception	{	
		
		double ind_class = (double) -1.0;
		if (TestClassifInstance_nom(inst,regle)) //Si la régle classifie cette instance nominale
				ind_class = regle.Rule_indClassMaj;		
		return ind_class ;		
	}
	
	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote pondéré
	 * @param inst : L'instance à classifier
	 * @param ensemble_regles: Ensemble de régles
	 * @param nb_class : Nombre de classes
	 * @return L'indice de la classe prédite pour l'instance inst
	 */
	public double classify_Instance_nom_VotePond(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class) throws Exception	
	{
		//Calendar calendar = Calendar.getInstance(); SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS"); 
		//System.err.println("\t"+sdf.format(calendar.getTime()));
		
		//Selection selon le vote pondéré		
		double [] tab_ponderation = new double [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_ponderation[i] = (double) 0.0;
		
		// Parcourir les régles de classification
		boolean etat_classement=false;
		int tmp_ind_class;
		for(int i=0; i<ensemble_regles.size(); i++)
		{		
			if ( TestClassifInstance_nom(inst, ensemble_regles.get(i)) ) //Si une régle classifie cette instance nominale
			{
				etat_classement=true;
				//System.out.println(sdf.format(calendar.getTime())+"\t"+ensemble_regles.get(i).affich_nom_rule(true));
				tmp_ind_class = (int) ensemble_regles.get(i).Rule_indClassMaj;
				//System.out.println("* tmp_ind_class: "+tmp_ind_class);
				tab_ponderation[tmp_ind_class] = tab_ponderation[tmp_ind_class] + ensemble_regles.get(i).Rule_Ponderation;
				//System.out.println("* tab_vote["+tmp_ind_class+"]: "+tab_ponderation[tmp_ind_class]);
			}
		}
		
		double ind_class;
		
		if(etat_classement)
		{
			//Rechercher l'indice de la classe la plus pondérée
			double tmp_ponderation;
			tmp_ponderation = -1;
			tmp_ind_class = -1;
			for(int i=0 ; i<nb_class ; i++)
				if(tab_ponderation[i] > tmp_ponderation)
				{
					tmp_ponderation = tab_ponderation[i];
					tmp_ind_class = i;
				}	
			ind_class = (double) tmp_ind_class;
		}
		else
			ind_class = (double) -1.0;
		
		//System.out.println("=====> ind_class: "+ind_class+"\n \t END \n\n\n");
		return ind_class ;			
	}
	
	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote majoritaire
	 * @param inst : L'instance à classifier
	 * @param ensemble_regles: Ensemble de régles
	 * @param nb_class : Nombre de classes
	 * @return L'indice de la classe prédite pour l'instance inst
	 */
	public double classify_Instance_nom_VoteMaj(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class) throws Exception 
	{	
		//Calendar calendar = Calendar.getInstance(); SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS"); 
		//System.err.println("\t"+sdf.format(calendar.getTime()));
			
		//Selection selon le vote majoré		
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		// Parcourir les régles de classification
		boolean etat_classement=false;
		int tmp_ind_class;
		for(int i=0; i<ensemble_regles.size(); i++)
		{		
			if (TestClassifInstance_nom(inst , ensemble_regles.get(i))) //Si une régle classifie cette instance nominale
			{
				etat_classement=true;
				//System.out.println(sdf.format(calendar.getTime())+"\t"+ensemble_regles.get(i).affich_nom_rule(true));
				tmp_ind_class = (int) ensemble_regles.get(i).Rule_indClassMaj; 
				//System.out.println("* tmp_ind_class: "+tmp_ind_class);
				tab_vote[tmp_ind_class] ++; 
				//System.out.println("* tab_vote["+tmp_ind_class+"]: "+tab_vote[tmp_ind_class]);
			}
		}
		
		double ind_class;
		
		if(etat_classement)
		{
			//Rechercher l'indice de la classe la plus votée
			double tmp_vote;
			tmp_vote = -1;
			tmp_ind_class = -1;
			for(int i=0 ; i<nb_class ; i++)
				if(tab_vote[i] > tmp_vote)
				{
					tmp_vote = tab_vote[i];
					tmp_ind_class = i;
				}	
			ind_class = (double) tmp_ind_class; //System.out.println("* ind_class: "+ind_class+"\n \t END \n\n\n");			
		}
		else
			ind_class = (double) -1.0;
		
		//System.out.println("=====> ind_class: "+ind_class+"\n \t END \n\n\n");
		return ind_class ;			
	}

	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote à la pluralité
	 * @param inst : L'instance à classifier
	 * @param ensemble_regles: Ensemble de régles
	 * @param nb_class : Nombre de classes
	 * @return L'indice de la classe prédite pour l'instance inst
	 */
	public double classify_Instance_nom_VotePlur(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class) throws Exception 
	{
		//Calendar calendar = Calendar.getInstance(); SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS"); 
		//System.err.println("\t"+sdf.format(calendar.getTime()));
			
		//Selection selon le vote majoré		
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		// Parcourir les régles de classification
		boolean etat_classement=false;
		int tmp_ind_class;
		for(int i=0; i<ensemble_regles.size(); i++)
		{		
			if (TestClassifInstance_nom(inst , ensemble_regles.get(i))) //Si une régle classifie cette instance nominale
			{
				etat_classement=true;
				//System.out.println(sdf.format(calendar.getTime())+"\t"+ensemble_regles.get(i).affich_nom_rule(true));
				tmp_ind_class = (int) ensemble_regles.get(i).Rule_indClassMaj; 
				//System.out.println("* tmp_ind_class: "+tmp_ind_class);
				tab_vote[tmp_ind_class] ++; 
				//System.out.println("* tab_vote["+tmp_ind_class+"]: "+tab_vote[tmp_ind_class]);
			}
		}
		
		double ind_class;
		
		if(etat_classement)
		{
			//Rechercher l'indice de la classe la plus votée
			double tmp_vote;
			tmp_vote = -1;
			tmp_ind_class = -1;
			for(int i=0 ; i<nb_class ; i++)
				if(tab_vote[i] > tmp_vote)
				{
					tmp_vote = tab_vote[i];
					tmp_ind_class = i;
				}
			
			if( tab_vote[tmp_ind_class] > (ensemble_regles.size()/2) )
				ind_class = (double) tmp_ind_class; 
			else
				ind_class = (double) -1.0;
		}
		else
			ind_class = (double) -1.0;
		
		//System.out.println("=====> ind_class: "+ind_class+"\n \t END \n\n\n");
		return ind_class ;			
	}

	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote à la pluralité avec un seuil variable
	 * @param inst : L'instance à classifier
	 * @param ensemble_regles: Ensemble de régles
	 * @param nb_class : Nombre de classes
	 * @param PlurMin : Seuil minimum pour la pluralité d'un vote acceptable
	 * @return L'indice de la classe prédite
	 */
	public double classify_Instance_nom_VotePlurMin(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, double PlurMin) 
			throws Exception 
	{	
		//Calendar calendar = Calendar.getInstance(); SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS"); 
		//System.err.println("\t"+sdf.format(calendar.getTime()));
		
		//Selection selon le vote majoré		
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		// Parcourir les régles de classification
		boolean etat_classement=false;
		int tmp_ind_class;
		for(int i=0; i<ensemble_regles.size(); i++)
		{		
			if (TestClassifInstance_nom(inst,ensemble_regles.get(i))) //Si une régle classifie cette instance nominale
			{
				etat_classement=true;
				//System.out.println(sdf.format(calendar.getTime())+"\t"+ensemble_regles.get(i).affich_nom_rule(true));
				tmp_ind_class = (int) ensemble_regles.get(i).Rule_indClassMaj; 
				//System.out.println("* tmp_ind_class: "+tmp_ind_class);
				tab_vote[tmp_ind_class] ++; 
				//System.out.println("* tab_vote["+tmp_ind_class+"]: "+tab_vote[tmp_ind_class]);
			}
		}
		
		double ind_class;
		
		if(etat_classement)
		{
			//Rechercher l'indice de la classe la plus votée
			double tmp_vote;
			tmp_vote = -1;
			tmp_ind_class = -1;
			for(int i=0 ; i<nb_class ; i++)
				if(tab_vote[i] > tmp_vote)
				{
					tmp_vote = tab_vote[i];
					tmp_ind_class = i;
				}
			double conv_PlurMin = (double) PlurMin*ensemble_regles.size() / 100;
			/*System.out.println(
					"* tab_vote[tmp_ind_class] : "+tab_vote[tmp_ind_class]
					+"\t ensemble_regles.size() : "+ensemble_regles.size()
					+"\t PlurMin : "+PlurMin
					+"\t conv_PlurMin : "+conv_PlurMin);
					*/
			if( tab_vote[tmp_ind_class] >= conv_PlurMin ) 
				ind_class = (double) tmp_ind_class; 
			else
				ind_class = (double) -1.0;
		}
		else
			ind_class = (double) -1.0;
		
		//System.out.println("* ind_class: "+ind_class+"\n \t END \n\n\n");
		return ind_class ;			
	}

	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote majoritaire par des classifieurs 
	 * qui dépassent un seuil donné de pondération (vote notoire)
	 * @param inst : L'instance à classifier
	 * @param ensemble_regles: Ensemble de régles
	 * @param nb_class : Nombre de classes
	 * @param LowPondVoter : Le seuil minimum du poids des classifieurs participant dans le vote
	 * @return L'indice de la classe prédite 
	 */
	public double classify_Instance_nom_VoteLowPondVoter(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, double LowPondVoter) 
			throws Exception 
	{	
		//Calendar calendar = Calendar.getInstance(); SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS"); 
		//System.err.println("\t"+sdf.format(calendar.getTime()));
				
		// Création d'un tableau pour vote majoritaire
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		//double conv_LowPondVoter = (double) LowPondVoter*ensemble_regles.size() / 100;
		double conv_LowPondVoter = (double) LowPondVoter / 100; 
		//System.out.println(sdf.format(calendar.getTime())+"\t conv_LowPondVoter = "+conv_LowPondVoter);
		
		// Parcourir les régles de classification
		boolean etat_classement=false;
		int tmp_ind_class;
		for(int i=0; i<ensemble_regles.size(); i++)
		{	
			//System.out.println(sdf.format(calendar.getTime())+"\t"+ensemble_regles.get(i).affich_nom_rule(true));
			//Si une régle classifie cette instance nominale
			if (ensemble_regles.get(i).getRule_Ponderation()>= conv_LowPondVoter && TestClassifInstance_nom(inst,ensemble_regles.get(i))) 
			{
				etat_classement=true;
				tmp_ind_class = (int) ensemble_regles.get(i).Rule_indClassMaj; 
				//System.out.println("* tmp_ind_class: "+tmp_ind_class);
				tab_vote[tmp_ind_class] ++; 
				//System.out.println("* tab_vote["+tmp_ind_class+"]: "+tab_vote[tmp_ind_class]);
			}
		}
		
		double ind_class;
		
		if(etat_classement)
		{
			//Rechercher l'indice de la classe la plus votée
			double tmp_vote;
			tmp_vote = -1;
			tmp_ind_class = -1;
			for(int i=0 ; i<nb_class ; i++)
				if(tab_vote[i] > tmp_vote)
				{
					tmp_vote = tab_vote[i];
					tmp_ind_class = i;
				}
			ind_class = (double) tmp_ind_class; 
		}
		else
			ind_class = (double) -1.0;
				
		//System.out.println("* ind_class: "+ind_class+"\n \t END \n\n\n");

		return ind_class ;			
	}

	/**
	 * Cette fonction permet de classifier une instance nominale 
	 * par une technique de vote pondéré à partir d'un nombre bien
	 * déterminé du meilleurs classifieurs
	 * @param inst : L'instance à classifier
	 * @param ensemble_regles: Ensemble de régles
	 * @param nb_class : Nombre de classes
	 * @param MaxBestVoterS : Nombre des classifieurs
	 * @return L'indice de la classe prédite pour l'instance inst
	 */
	public double classify_Instance_nom_VoteNumBestVoterS(Instance inst, ArrayList<Classification_Rule> ensemble_regles, int nb_class, int MaxBestVoterS) throws Exception {	
		
		//Calendar calendar = Calendar.getInstance(); SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS"); 
		//System.err.println("\t"+sdf.format(calendar.getTime()));
			
		// Creer un tableau à deux dimensions pour contenir les indices des classifieurs et leurs poids
		double [][] ens_reg_sorted = new double [ensemble_regles.size()][2];
		// Initialiser ens_reg_sorted[][] par les indices des classifieurs et leurs poids
		for(int i=0; i<ensemble_regles.size(); i++)
		{
			ens_reg_sorted[i][0] = (double) i;
			ens_reg_sorted[i][1] = (double) ensemble_regles.get(i).getRule_Ponderation();
			//System.out.println("ens_reg_sorted["+i+"][0] = "+ens_reg_sorted[i][0]+ "\t ens_reg_sorted["+i+"][1] = "+ ens_reg_sorted[i][1]);
		}
				
		// Trier ens_reg_sorted[][] par ordre décroissant		
		for(int i=0; i<ensemble_regles.size(); i++)
			for(int j=i+1; j<ensemble_regles.size(); j++)
			{
				if (ens_reg_sorted[j][1] > ens_reg_sorted[i][1])
				{
					double tmp_ind, tmp_pond;
					tmp_ind = ens_reg_sorted[j][0];
					tmp_pond = ens_reg_sorted[j][1];

					ens_reg_sorted[j][0] = i;
					ens_reg_sorted[j][1] = ens_reg_sorted[i][1];

					ens_reg_sorted[i][0] = tmp_ind;
					ens_reg_sorted[i][1] = tmp_pond;	
				}
			}
		
		 /*for(int i=0; i<ensemble_regles.size(); i++) 
			 System.out.println("ens_reg_sorted["+i+"][0] = "+ens_reg_sorted[i][0]+ "\t ens_reg_sorted["+i+"][1] = "+ ens_reg_sorted[i][1]);
			 */
		
		// Selection selon le vote majoré		
		int [] tab_vote = new int [nb_class];
		for(int i=0; i<nb_class; i++)
			tab_vote[i] = (int) 0;
		
		// Parcourir les régles de classification
		boolean etat_classement=false;
		int tmp_ind_class;
		for(int i=0 ; i<ensemble_regles.size() && i<=(MaxBestVoterS-1) ; i++)
		{		
			if (TestClassifInstance_nom(inst , ensemble_regles.get((int) ens_reg_sorted[i][0]))) //Si une régle classifie cette instance nominale
			{
				etat_classement=true;
				tmp_ind_class = (int) ensemble_regles.get((int) ens_reg_sorted[i][0]).Rule_indClassMaj; 
				//System.out.println("* tmp_ind_class: "+tmp_ind_class);
				tab_vote[tmp_ind_class] ++; 
				//System.out.println("* tab_vote["+tmp_ind_class+"]: "+tab_vote[tmp_ind_class]);
			}
		}
		
		// Rechercher l'indice de la classe la plus votée		
		double ind_class;
		if(etat_classement)
		{
			double tmp_vote;
			tmp_vote = -1;
			tmp_ind_class = -1;
			for(int i=0 ; i<nb_class ; i++)
				if(tab_vote[i] > tmp_vote)
				{
					tmp_vote = tab_vote[i];
					tmp_ind_class = i;
				}	
			ind_class = (double) tmp_ind_class; 
		}
		else
			ind_class = (double) -1.0;
		
		//System.out.println("* ind_class: "+ind_class+"\n \t END \n\n\n");
		return ind_class ;				
	}
	
	/**
	 * Parcourir les attributs nominaux de 'inst' et les comparer à ceux de 'regle'.
	 * Le but est de voir si 'regle' est applicable pour prédire la classe de 'inst'
	 * Ignorer le dernier attribut qui represente la classe (inst.numAttributes()-1)
	 * Verifer que le i iéme attribut de la régle est different de '-null-' 
 	 * aussi différent de la i éme attribut de l'instance nominale à classifier
	 * @param regle: La régle de classification
	 * @param inst: l'istance à classifier
	 * @return True ou False
	 */
	public boolean TestClassifInstance_nom(Instance inst , Classification_Rule regle) throws Exception 
	{
		////System.out.println("\n\n\n");
		//System.out.println ("La régle de classification : " + regle.affich_nom_rule(true));
		//System.out.println("L'instance à prédire sa classe: " + inst.toString()+"\t --> \t");
		
		  
		String nl = "-null-";
		for (int i=0 ; i<inst.numAttributes()-1 ; i++)
			{
			if(regle.Rule_Attr.get(i).compareTo(nl)!=0 
			&& regle.Rule_Attr.get(i).compareTo(inst.stringValue(i))!=0)
			{
				//System.out.println("false");
				return false;
			}
		}
		//System.out.println("true");
		return true;
	}

}
