package liblaries.neuralNetwork.learning;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_context;
import org.jocl.cl_mem;

public class LearningSeqence{
	public float[] inputs;
	public float[] outputs;
	
	public cl_mem inputsCL=null;
	public cl_mem outputsCL=null;
	
	public LearningSeqence(){}
	public LearningSeqence(float[] Wejœcia, float[] Wyjœcia){
		inputs=Wejœcia;
		outputs=Wyjœcia;
	}
	
	public void initializeCL(cl_context context) {
		inputsCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*inputs.length, Pointer.to(inputs), null);
		outputsCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*outputs.length, Pointer.to(outputs), null);
	}
	public void clearCL() {
		CL.clReleaseMemObject(inputsCL);
		CL.clReleaseMemObject(outputsCL);
	}
}