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
package org.ml4j.nn.components.manytoone;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.manytoone.base.ManyToOneDirectedComponentBase;
import org.ml4j.nn.neurons.DummyNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyManyToOneDirectedComponent extends ManyToOneDirectedComponentBase<DummyManyToOneDirectedComponentActivation> implements ManyToOneDirectedComponent<DummyManyToOneDirectedComponentActivation> {

	public DummyManyToOneDirectedComponent(PathCombinationStrategy pathCombinationStrategy) {
		super(pathCombinationStrategy);
	}

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyManyToOneDirectedComponent.class);
	
	/**s
	 * Serialization id.
	 */
	private static final long serialVersionUID = -7049642040068320620L;

	@Override
	public DummyManyToOneDirectedComponentActivation forwardPropagate(List<NeuronsActivation> neuronActivations,
			DirectedComponentsContext context) {
		LOGGER.debug("Mock combining multiple neurons activations into a single output neurons activation") ;
		int outputNeuronCount;
		if (pathCombinationStrategy == PathCombinationStrategy.ADDITION) {
			outputNeuronCount = neuronActivations.get(0).getFeatureCount();
		} else if (pathCombinationStrategy == PathCombinationStrategy.FILTER_CONCAT){
			outputNeuronCount = neuronActivations.stream().mapToInt(a -> a.getFeatureCount()).sum();
		} else {
			throw new UnsupportedOperationException("Path combination strategy of:" + pathCombinationStrategy + " not supported");
		}
		Neurons mockOutputNeurons = new Neurons(outputNeuronCount, false);
		NeuronsActivation mockOutput = new DummyNeuronsActivation(mockOutputNeurons, neuronActivations.get(0).getFeatureOrientation(), 
				neuronActivations.get(0).getExampleCount());
		return new DummyManyToOneDirectedComponentActivation(neuronActivations.size(), mockOutput);
	}

	@Override
	public ManyToOneDirectedComponent<DummyManyToOneDirectedComponentActivation> dup() {
		return new DummyManyToOneDirectedComponent(pathCombinationStrategy);
	}
	
	@Override
	public Optional<NeuronsActivationFeatureOrientation> optimisedFor() {
		// TODO THUR
		return Optional.empty();
	}

	@Override
	public List<NeuronsActivationFeatureOrientation> supports() {
		// TODO THUR
		return Arrays.asList(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	}
}
