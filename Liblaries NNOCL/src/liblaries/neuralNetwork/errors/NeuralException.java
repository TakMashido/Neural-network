package liblaries.neuralNetwork.errors;

public class NeuralException extends RuntimeException{
	private static final long serialVersionUID = -5734111472592152319L;
	
	//error ID's
	public static final int invalidClear=0; 				//Throwed when you try to clear network data not necessery for GPU calculation without inintiate openCL
	public static final int learningMemChange=1;			//Throwed when you try to change eg. weights of network during learning. First stop learning
	public static final int invalidOutputSize=2;			//Throwed when outputs number of learning sequence is diffrend than network outputs number
	public static final int learningInProgress=3;			//Throwed when you try start second learning when first is not finished, or set teacher's network durnig learning, or change network structure during learning
	public static final int invalidInputSize=4;				//Throwed when size of given input(or in LS) is different then network input number
	public static final int notSupportOpenCL=5;				//Throwed when you try use openCL on network which don't support this
	public static final int notSupportMultiThreading=6;		//Throwed when you try to set more threads on network witch didn't support it
	public static final int multiThreadingError=7;			//Throwed when worker thread gets interrupted or Cyclic barrier throws error. If this occur this is propably some liblaries bug. Please report an issue on github with stack trace.
	public static final int noLearningSequence=8;			//Throwed when you try to start learning network without learning sequence seted
	private int errorID;
	
	public NeuralException(int errorID) {
		super(errorIDToString(errorID));
		this.errorID=errorID;
	}
	public static String errorIDToString(int errorID){
		switch(errorID) {
		case invalidClear:return "First initiate openCL";
		case learningMemChange:return "Can't change data. Learning in progress";
		case invalidOutputSize:return "Invalid output number in LS";
		case learningInProgress:return "Please stop learning before performinig this operation";
		case invalidInputSize: return "Invalid input number";
		case notSupportOpenCL: return "OpenCL not supported for this operation";
		case notSupportMultiThreading: return "This network didn't support multithreading";
		case multiThreadingError: return "Error ocured in worker thread";
		case noLearningSequence: return "Please set learning sequence";
		default :return "Unknow error";
		}
	}
	
	public int getIntErrorID() {
		return errorID;
	}
}