package org.ml4j.nn.activationfunctions;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationContextImpl;

public class LeakyReluActivationFunctionContext extends NeuronsActivationContextImpl implements NeuronsActivationContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private float alpha;

	public LeakyReluActivationFunctionContext(MatrixFactory matrixFactory, boolean isTrainingContext, float alpha) {
		super(matrixFactory, isTrainingContext);
		this.alpha = alpha;
	}

	public float getAlpha() {
		return alpha;
	}	
}
