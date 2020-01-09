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
package org.ml4j.nn.components.manytomany;

import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentActivationLifecycle;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.manytomany.base.DirectedComponentChainBatchActivationBase;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDirectedComponentChainBatchActivation extends DirectedComponentChainBatchActivationBase implements DefaultDirectedComponentChainBatchActivation{

	
	public DummyDirectedComponentChainBatchActivation(List<DefaultDirectedComponentChainActivation> activations) {
		super(activations);
	}

	@Override
	public DirectedComponentGradient<List<NeuronsActivation>> backPropagate(
			DirectedComponentGradient<List<NeuronsActivation>> gradient) {
		return gradient;
	}

	@Override
	public List<? extends ChainableDirectedComponentActivation<List<NeuronsActivation>>> decompose() {
		// TODO ML
		return null;
	}
	
	@Override
	public void close(DirectedComponentActivationLifecycle arg0) {
		// No-op
	}
}
