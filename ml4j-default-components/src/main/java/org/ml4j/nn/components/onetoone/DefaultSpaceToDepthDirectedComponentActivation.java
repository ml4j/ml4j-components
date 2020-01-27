package org.ml4j.nn.components.onetoone;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.DirectedComponentActivationLifecycle;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.base.DefaultChainableDirectedComponentActivationBase;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultSpaceToDepthDirectedComponentActivation extends DefaultChainableDirectedComponentActivationBase<DefaultSpaceToDepthDirectedComponent> {

	public DefaultSpaceToDepthDirectedComponentActivation(DefaultSpaceToDepthDirectedComponent originatingComponent,
			NeuronsActivation output) {
		super(originatingComponent, output);
	}

	@Override
	public List<? extends DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public void close(DirectedComponentActivationLifecycle completedLifeCycleStage) {
		// TODO
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {
		// TODO Auto-generated method stub
		return null;
	}

}
