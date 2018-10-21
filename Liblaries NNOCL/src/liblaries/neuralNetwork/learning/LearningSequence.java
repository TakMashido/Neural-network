package liblaries.neuralNetwork.learning;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_mem;

public class LearningSequence{
	public float[] inputs;
	public float[] outputs;
	
	public cl_mem inputsCL=null;				//with bias
	public cl_mem outputsCL=null;				//with bias
	
	public LearningSequence(){}
	public LearningSequence(float[] inputs){
		this.inputs=inputs;
	}
	public LearningSequence(float[] inputs, float[] outputs){
		this.inputs=inputs;
		this.outputs=outputs;
	}
	
	public static LearningSequence[] create(float[][] inputs) {
		return create(inputs,null);
	}
	public static LearningSequence[] create(float[][] inputs, float[][] outputs){
		LearningSequence[] ls=new LearningSequence[inputs.length];
		
		if(outputs!=null)
			for(int i=0;i<ls.length;i++) {
				ls[i]=new LearningSequence(inputs[i],outputs[i]);
			}
		else
			for(int i=0;i<ls.length;i++) {
				ls[i]=new LearningSequence(inputs[i]);
			}
		
		return ls;
	}
	
	public void initializeCL(cl_context context,cl_command_queue commandQueue) {
		inputsCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*(inputs.length+1), Pointer.to(new float[] {1}), null);
		CL.clEnqueueWriteBuffer(commandQueue, inputsCL, CL.CL_TRUE, Sizeof.cl_float, Sizeof.cl_float*inputs.length, Pointer.to(inputs), 0, null, null);
		if(outputs!=null) {
			outputsCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*(outputs.length+1), Pointer.to(new float[] {1}), null);
			CL.clEnqueueWriteBuffer(commandQueue, outputsCL, CL.CL_TRUE, Sizeof.cl_float, Sizeof.cl_float*inputs.length, Pointer.to(outputs), 0, null, null);
		}
	}
	public void clearCL() {
		CL.clReleaseMemObject(inputsCL);
		if(outputsCL!=null)CL.clReleaseMemObject(outputsCL);
	}
}