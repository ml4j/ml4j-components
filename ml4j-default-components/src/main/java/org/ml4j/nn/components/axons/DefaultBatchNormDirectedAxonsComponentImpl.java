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
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DefaultBatchNormDirectedAxonsComponentImpl<L extends Neurons, R extends Neurons> extends DirectedAxonsComponentBase<L, R, Axons<? extends L, ? extends R, ?>> 
	implements BatchNormDirectedAxonsComponent<L, R> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDirectedAxonsComponentImpl.class);
	/**
	 * Defaut serialization id;
	 */
	private static final long serialVersionUID = 1L;

	
	public DefaultBatchNormDirectedAxonsComponentImpl(Axons<? extends L, ? extends R, ?> axons) {
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
	public BatchNormDirectedAxonsComponent<L, R> dup() {
		return new DefaultBatchNormDirectedAxonsComponentImpl<>(axons.dup());
	}

	@Override
	public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation neuronsActivation, AxonsContext axonsContext) {
		LOGGER.debug("Forward propagating through DefaultBatchNormDirectedAxonsComponentImpl");
		
		AxonsActivation axonsActivation = axons.pushLeftToRight(neuronsActivation, null, axonsContext);
		
		return new DefaultBatchNormDirectedAxonsComponentActivationImpl(axonsContext, this, axonsActivation);
	}

}
