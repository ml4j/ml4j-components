/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.components.onetoone;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.DirectedComponentActivationLifecycle;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainActivationBase;
import org.ml4j.nn.neurons.DummyNeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Mock implementation of DefaultDirectedComponentChainActivation, and provides logic to back propagate gradient through the activation.
 * 
 * Encapsulates the mock activations from a forward propagation through a DefaultDirectedComponentChain.
 * 
 * @author Michael Lavelle
 */
public class DummyDefaultDirectedComponentChainActivation extends DefaultDirectedComponentChainActivationBase<DefaultDirectedComponentChain> implements DefaultDirectedComponentChainActivation {
	
	public DummyDefaultDirectedComponentChainActivation(DefaultDirectedComponentChain componentChain, List<DefaultChainableDirectedComponentActivation> activations) {
		super(componentChain, activations, activations.get(activations.size() - 1).getOutput());
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		return new DirectedComponentGradientImpl<NeuronsActivation>(new DummyNeuronsActivation(originatingComponent.getInputNeurons(), gradient.getOutput().getFeatureOrientation(), gradient.getOutput().getExampleCount()));
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(this);
	}
	
	@Override
	public void close(DirectedComponentActivationLifecycle arg0) {
		// No-op
	}

}
