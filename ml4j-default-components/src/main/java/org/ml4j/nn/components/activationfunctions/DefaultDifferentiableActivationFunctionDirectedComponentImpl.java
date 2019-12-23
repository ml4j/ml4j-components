package org.ml4j.nn.components.activationfunctions;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;

public class DefaultDifferentiableActivationFunctionDirectedComponentImpl extends DifferentiableActivationFunctionComponentBase implements DifferentiableActivationFunctionComponent {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
		
	public DefaultDifferentiableActivationFunctionDirectedComponentImpl(Neurons neurons, DifferentiableActivationFunction activationFunction) {
		super(neurons, activationFunction);
	}

	@Override
	public DifferentiableActivationFunctionComponentActivation forwardPropagate(NeuronsActivation input,
			NeuronsActivationContext context) {
		if (input.getFeatureCount() != getInputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalArgumentException("Expected input neurons activation to span " + getInputNeurons().getNeuronCountExcludingBias() + " features");
		}
		return new DefaultDifferentiableActivationFunctionComponentActivationImpl(this, input, activationFunction.activate(input, context).getOutput());
	}

	@Override
	public DifferentiableActivationFunctionComponent dup() {
		return new DefaultDifferentiableActivationFunctionDirectedComponentImpl(neurons, activationFunction);
	}
}
