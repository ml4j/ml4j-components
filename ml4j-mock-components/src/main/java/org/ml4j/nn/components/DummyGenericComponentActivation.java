package org.ml4j.nn.components;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyGenericComponentActivation implements DefaultChainableDirectedComponentActivation {

	private NeuronsActivation input;
	private NeuronsActivation output;
	
	public DummyGenericComponentActivation(NeuronsActivation input, NeuronsActivation output) {
		this.input = input;
		this.output = output;
	}
	@Override
	public void close(DirectedComponentActivationLifecycle completedLifeCycleStage) {
		// No-op
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {
		return new DirectedComponentGradientImpl<>(outerGradient.getTotalTrainableAxonsGradients(), null, input);
	}

	@Override
	public NeuronsActivation getOutput() {
		return output;
	}

	@Override
	public List<? extends DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(this);
	}
}
