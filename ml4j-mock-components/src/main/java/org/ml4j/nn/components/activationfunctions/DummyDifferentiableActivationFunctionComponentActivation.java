package org.ml4j.nn.components.activationfunctions;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionActivationBase;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Dummy implementation of DifferentiableActivationFunctionActivation, encapsulating the activations from a DummyDifferentiableActivationFunction
 * and providing the logic required in order to back propagate gradients back through the activations.
 * 
 * @author Michael Lavelle
 */
public class DummyDifferentiableActivationFunctionComponentActivation extends DifferentiableActivationFunctionActivationBase implements DifferentiableActivationFunctionActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyDifferentiableActivationFunctionComponentActivation.class);
		
	/**
	 * @param input The input to the DummyDifferentiableActivationFunction
	 * @param output The output from the DummyDifferentiableActivationFunction
	 */
	public DummyDifferentiableActivationFunctionComponentActivation(NeuronsActivation input, NeuronsActivation output) {
		super(input, output);
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		LOGGER.debug("Back propagating gradient through DummyDifferentiableActivationFunctionActivation");
		return gradient;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient costFunctionGradient) {
		LOGGER.debug("Back propagating cost function gradient through DummyDifferentiableActivationFunctionActivation");
		return new DirectedComponentGradientImpl<>(output);
	}

	@Override
	public DifferentiableActivationFunction getActivationFunction() {
		throw new UnsupportedOperationException("Not required for this mock implementation");
	}

}
