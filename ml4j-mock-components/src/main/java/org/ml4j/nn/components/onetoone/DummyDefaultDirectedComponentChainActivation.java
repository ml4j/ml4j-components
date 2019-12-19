package org.ml4j.nn.components.onetoone;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDefaultDirectedComponentChainActivation extends DefaultDirectedComponentChainActivationBase implements DefaultDirectedComponentChainActivation {
	
	public DummyDefaultDirectedComponentChainActivation(NeuronsActivation output) {
		super(output);
	}
	
	@Override
	public List<DefaultChainableDirectedComponentActivation> getActivations() {
		return Arrays.asList(this);
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		return gradient;
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(this);
	}

}
