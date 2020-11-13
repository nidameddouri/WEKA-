package weka.classifiers.fca;
import weka.core.*;

import java.util.ArrayList;

//import java_cup.parser;

import java.awt.Color;
import java.awt.*;

/**
 * @author Meddouri Nida (nida.meddouri@gmail.com)
 * revision: 20160823  
 */

public class Classification_Rule  {
	
	//public ArrayList <String> Rule_Attr_Name; // La liste des noms d'attributs
	public ArrayList <String> Rule_Attr; // La liste des VALEURS d'attributs
	public double Rule_indClassMaj; // indice de la classe majoritaire.
	public String Rule_ClassMaj; // Classe majoritaire.
	public double Rule_Ponderation;	// La valuer de la pondération.  

	//throws Exception 

// Un constructeur null d'une régle
	public Classification_Rule () {
		//this.Rule_Attr_Name	= new ArrayList <String>();
		this.Rule_Attr 			= new ArrayList <String>();
		this.Rule_indClassMaj 	= (double) -1.0;
		this.Rule_ClassMaj 		= "";
		this.Rule_Ponderation 	= (double) -1.0;	
	}
	
	// Un constructeur à partir de la liste des attributs, indice et classe majoritaire.	
	public Classification_Rule(ArrayList<String> attributes, double ind_CM, String CM){
		this.Rule_Attr 			= new ArrayList <String>();
		this.Rule_indClassMaj 	= (double) ind_CM;
		this.Rule_ClassMaj 		= CM;
		
		for(int i=0; i< attributes.size();i++)	
			 this.Rule_Attr.add(attributes.get(i));
	}
	
	// Un constructeur à partir de la liste des attribut, indice et classe majoritaire et pondération.	
	public Classification_Rule(ArrayList<String> attributes, double ind_CM, String CM, double ponderation){
		this.Rule_Attr 			= new ArrayList <String>();
		this.Rule_indClassMaj	= (double) ind_CM;
		this.Rule_ClassMaj 		= CM;
		this.Rule_Ponderation 	= ponderation;
		
		for(int i=0; i< attributes.size();i++)		
			this.Rule_Attr.add(attributes.get(i));		
	}
	
	// Copier la régle this dans une autre en retour
	public Classification_Rule copieRules ()	{
		Classification_Rule ob 	= new Classification_Rule ();
		ob.Rule_Attr 			= this.getRule_Attr();
		ob.Rule_indClassMaj 	= this.getRule_indClassMaj();
		ob.Rule_ClassMaj 		= this.getRule_ClassMaj();
		ob.Rule_Ponderation 	= this.getRule_Ponderation();	
	    return ob;		
	}
	
/*	public String toString  (ArrayList <String> rule){
		String chaine=new String ("IF "); 
 
       	// Pour afficher les attributs sous format textuel
	    for (int g=0; g<this.Rule_Attr.size();g++)
	    	chaine =chaine +rule.get(g)+ " AND";   	    	    
	    
	    chaine =chaine +" ";
	    chaine =chaine +" THEN  "+ this.getRule_indClassMaj();
	    chaine =chaine +" WITH Ponderation =  "+ Utils.roundDouble(this.getRule_Ponderation(),2);
	       
		return chaine;
	}
*/		
	// Affichage nominal de la prémisse de la régle. Si AfficheCM est vraie, on associe une classe majoritaire sinon sans classe
	public String affich_nom_rule (boolean AfficheCM){
		
		String chaine = new String ("IF ");
		for (int j=0; j< this.Rule_Attr.size(); j++)
			if(j!=this.Rule_Attr.size()-1)
     	   		chaine = chaine+this.Rule_Attr.get(j)+" AND ";
			else
				chaine = chaine+this.Rule_Attr.get(j);
		
	     chaine = chaine+"  ";
	     if (AfficheCM)
		        chaine = chaine + " THEN  " + this.getRule_ClassMaj() +"\t Indice classe: "+this.getRule_indClassMaj()+"  ";
	          
	        chaine = chaine + "\t Pondération: " + Utils.roundDouble(this.getRule_Ponderation(),2); 
	        
		return chaine;
	}
	
	// Affichage nominal de la prémisse de la régle.
	public String affich_nom_rules_ForDiversity (){
		String chaine=new String ("");
		for (int j=0; j< this.Rule_Attr.size()-1; j++)
     	   	chaine = chaine+this.Rule_Attr.get(j)+",";        	
		chaine = chaine+this.Rule_Attr.get(this.Rule_Attr.size()-1);  	     	   
		//chaine = chaine+",";	          
	    //chaine = chaine +Utils.roundDouble(this.getRule_Ponderatio(),2);	        
		return chaine;
	}
	
	// Comparer cette régle à une autre y compris la partie conclusion
	 public boolean isEqual (Classification_Rule rl){
		 for (int i=0 ; i< rl.getRule_Attr().size(); i++)
			 if ( ! this.Rule_Attr.get(i).equals(rl.getRule_Attr().get(i)))
				 return false;
		 
		 if ( this.Rule_indClassMaj != rl.getRule_indClassMaj())
			 return false;
		 
		 return true ;
		 }

	// Comparer cette régle à une autre sans la partie conclusion
	 public boolean isEqualwithoutCL (Classification_Rule rl){
		 for (int i=0 ; i< rl.getRule_Attr().size(); i++)
			 if ( ! this.Rule_Attr.get(i).equals(rl.getRule_Attr().get(i)))
				 return false;
		 return  true ;	 
 	
      }
	
	
	public void setRule_Ponderation(double Rule_Ponderatio) {
		this.Rule_Ponderation = Rule_Ponderatio;
		}

	public void setRule_indClassMaj( double Rule_indClassMa){		
		this.Rule_indClassMaj = Rule_indClassMa;
		}

	public void setRule_ClassMaj( String Rule_ClassMa){		
		this.Rule_ClassMaj = Rule_ClassMa;
		}
	
	public void setRule_Attr(ArrayList<String> tab){
		this.Rule_Attr =new ArrayList <String>();
		for(int i=0; i< tab.size();i++)
			     this.Rule_Attr.add(tab.get(i));
		}


	public double getRule_Ponderation() 	{
		return Rule_Ponderation;
		}

	public ArrayList<String> getRule_Attr() {
		return Rule_Attr;
		}
	
	public  double getRule_indClassMaj(){
		return this.Rule_indClassMaj;
		}
	
	public  String getRule_ClassMaj()	{
		return this.Rule_ClassMaj;
		}
	
	}
