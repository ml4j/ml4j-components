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
package org.ml4j.nn.components.axons;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentBase;
import org.ml4j.nn.neurons.DummyNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyBatchNormDirectedAxonsComponent<L extends Neurons> extends DirectedAxonsComponentBase<L, L, Axons<L, L, ?>> 
	implements BatchNormDirectedAxonsComponent<L, Axons<L, L, ?>> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyDirectedAxonsComponent.class);
	/**
	 * Defaut serialization id;
	 */
	private static final long serialVersionUID = 1L;

	
	public DummyBatchNormDirectedAxonsComponent(Axons<L, L, ?> axons) {
		super(axons);
	}

	@Override
	public float getBetaForExponentiallyWeightedAverages() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureMeans() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureVariances() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void setExponentiallyWeightedAverageInputFeatureMeans(Matrix arg0) {
		throw new UnsupportedOperationException();		
	}

	@Override
	public void setExponentiallyWeightedAverageInputFeatureVariances(Matrix arg0) {
		throw new UnsupportedOperationException();		
	}

	@Override
	public BatchNormDirectedAxonsComponent<L, Axons<L, L, ?>> dup() {
		return new DummyBatchNormDirectedAxonsComponent<>(axons.dup());
	}

	@Override
	public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation neuronsActivation, AxonsContext axonsContext) {
		LOGGER.debug("Forward propagating through DummyDirectedAxonsComponent");
		return new DummyDirectedAxonsComponentActivation<>(this, new DummyNeuronsActivation(axons.getRightNeurons(), 
				neuronsActivation.getFeatureOrientation(), neuronsActivation.getExampleCount()));
	}

}
