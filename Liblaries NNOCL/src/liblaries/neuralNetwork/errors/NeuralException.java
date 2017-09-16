package liblaries.neuralNetwork.errors;

public class NeuralException extends RuntimeException{
	private static final long serialVersionUID = -5734111472592152319L;
	
	//error ID's								//I'll sort this in future
	public static final int invalidClear=0; 				//Throwed when you try to clear network data not necessery for GPU calculation without inintiate openCL
	public static final int learningMemChange=1;			//Throwed when you try to change eg. weights of network during learning. First stop learning
	public static final int invalidLS=2;					//Throwed when inputs/outputs number of learning sequence is diffrend than network inputs/outputs number
	public static final int learningInProgress=3;			//Throwed when you try start second learning when first is not finished, or set teacher's network durnig learning, or save network duriong learning
	public static final int invalidInputSize=4;				//Throwed when size of given input(or in LS) is different then network input number
	
	private int errorID;
	
	public NeuralException(int errorID) {
		super(errorIDToString(errorID));
		this.errorID=errorID;
	}
	public static String errorIDToString(int errorID){
		switch(errorID) {
		case invalidClear:return "First initiate openCL";
		case learningMemChange:return "Can't change data. Learning in progress";
		case invalidLS:return "Invalid input/output number in LS";
		case learningInProgress:return "Can't start new lerning when previus isn't finished";
		default :return "Unknow error";
		}
	}
	
	public int getIntErrorID() {
		return errorID;
	}
}