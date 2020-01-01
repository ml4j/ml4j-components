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
package org.ml4j.nn.components.activationfunctions;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;

/**
 * Default implementation of DifferentiableActivationFunctionComponent
 * 
 * @author Michael Lavelle
 *
 */
public class DefaultDifferentiableActivationFunctionComponentImpl extends DifferentiableActivationFunctionComponentBase implements DifferentiableActivationFunctionComponent {

	/**
	 * Generated serialization id.
	 */
	private static final long serialVersionUID = -6033017517698579773L;
	
	/**
	 * Default constructor
	 * 
	 * @param neurons The neurons at which the underlying activation function is applied.
	 * @param activationFunction The underlying activation function.
	 */
	public DefaultDifferentiableActivationFunctionComponentImpl(Neurons neurons, DifferentiableActivationFunction activationFunction){
		super(neurons, activationFunction);
	}

	@Override
	public DifferentiableActivationFunctionComponentActivation forwardPropagate(NeuronsActivation inputActivation,
			NeuronsActivationContext context) {
		DifferentiableActivationFunctionActivation activationFunctionActivation = activationFunction.activate(inputActivation, context);
		return new DefaultDifferentiableActivationFunctionComponentActivationImpl(this, activationFunctionActivation, context);
	}

	@Override
	public DifferentiableActivationFunctionComponent dup() {
		return new DefaultDifferentiableActivationFunctionComponentImpl(neurons, activationFunction);
	}
}
