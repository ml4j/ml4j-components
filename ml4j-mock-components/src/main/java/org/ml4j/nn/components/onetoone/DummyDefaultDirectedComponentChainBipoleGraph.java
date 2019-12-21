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

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBipoleGraphBase;
import org.ml4j.nn.neurons.DummyNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDefaultDirectedComponentChainBipoleGraph extends DefaultDirectedComponentChainBipoleGraphBase implements DefaultDirectedComponentBipoleGraph {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> parallelComponentChainsBatch;

	public DummyDefaultDirectedComponentChainBipoleGraph(Neurons inputNeurons, Neurons outputNeurons,
			DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> parallelComponentChainsBatch) {
		super(inputNeurons, outputNeurons, parallelComponentChainsBatch);
	}

	@Override
	public DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> getEdges() {
		throw new UnsupportedOperationException();
	}
	
	@Override
	public DefaultDirectedComponentBipoleGraphActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {
		return new DummyDefaultDirectedComponentBipoleGraphActivation(new DummyNeuronsActivation(getOutputNeurons(),
				neuronsActivation.getFeatureOrientation(), neuronsActivation.getExampleCount()));
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public DefaultDirectedComponentBipoleGraph dup() {
		return new DummyDefaultDirectedComponentChainBipoleGraph(inputNeurons, outputNeurons, parallelComponentChainsBatch.dup());
	}

}
