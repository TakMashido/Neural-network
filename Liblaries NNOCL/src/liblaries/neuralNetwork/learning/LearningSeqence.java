package liblaries.neuralNetwork.learning;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_mem;

public class LearningSeqence{
	public float[] inputs;
	public float[] outputs;
	
	public cl_mem inputsCL=null;				//with bias
	public cl_mem outputsCL=null;				//without bias
	
	public LearningSeqence(){}
	public LearningSeqence(float[] Wejœcia, float[] Wyjœcia){
		inputs=Wejœcia;
		outputs=Wyjœcia;
	}
	
	public void initializeCL(cl_context context,cl_command_queue commandQueue) {
		inputsCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*(inputs.length+1), Pointer.to(new float[] {1}), null);
		CL.clEnqueueWriteBuffer(commandQueue, inputsCL, CL.CL_TRUE, Sizeof.cl_float, Sizeof.cl_float*inputs.length, Pointer.to(inputs), 0, null, null);
		outputsCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*outputs.length, Pointer.to(outputs), null);
	}
	public void clearCL() {
		CL.clReleaseMemObject(inputsCL);
		CL.clReleaseMemObject(outputsCL);
	}
}