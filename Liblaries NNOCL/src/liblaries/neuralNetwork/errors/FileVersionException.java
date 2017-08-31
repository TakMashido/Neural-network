package liblaries.neuralNetwork.errors;

public class FileVersionException extends RuntimeException{
	private static final long serialVersionUID = -3395382788174892330L;
	
	public FileVersionException(String message) {
		super(message);
	}
}