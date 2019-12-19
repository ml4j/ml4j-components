package org.ml4j.nn.components.activationfunctions;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;

public class DummyDifferentiableActivationFunctionComponent extends DifferentiableActivationFunctionComponentBase implements DifferentiableActivationFunctionComponent {

	/**
	 * Generated serialization id.
	 */
	private static final long serialVersionUID = -6033017517698579773L;
	
	public DummyDifferentiableActivationFunctionComponent(DifferentiableActivationFunction activationFunction){
		super(activationFunction);
	}

	@Override
	public DifferentiableActivationFunctionActivation forwardPropagate(NeuronsActivation neuronsActivation,
			NeuronsActivationContext context) {
		return new DummyDifferentiableActivationFunctionComponentActivation(neuronsActivation, neuronsActivation);
	}

	@Override
	public DifferentiableActivationFunctionComponent dup() {
		return new DummyDifferentiableActivationFunctionComponent(activationFunction);
	}

}
