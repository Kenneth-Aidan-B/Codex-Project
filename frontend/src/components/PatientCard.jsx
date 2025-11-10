import React from 'react';
import PropTypes from 'prop-types';
import ProgressBar from './ProgressBar';
import './PatientCard.css';

const PatientCard = ({ record, prediction }) => {
  const { first_name: firstName, last_name: lastName, patient_id: patientId } = record;
  const probability = prediction?.probability ?? 0;
  const riskCategory = prediction?.risk_category ?? 'Unknown';
  const confidence = prediction?.confidence ?? 0;
  const explanation = prediction?.explanation?.top_features ?? [];

  return (
    <div className="patient-card">
      <header className="patient-card__header">
        <div>
          <h3>{firstName} {lastName}</h3>
          <p className="patient-card__subtitle">ID: {patientId || 'N/A'}</p>
        </div>
        <div className="patient-card__risk">{riskCategory}</div>
      </header>
      <section className="patient-card__body">
        <ProgressBar value={probability} label={`Risk: ${riskCategory}`} />
        <p className="patient-card__confidence">Confidence: {(confidence * 100).toFixed(1)}%</p>
        <div className="patient-card__metrics">
          <h4>Key Features</h4>
          <ul>
            {explanation.map((item) => (
              <li key={`${item.name}-${item.value}`}>
                <span>{item.name}</span>
                <span className="patient-card__metric-value">{item.value} ({item.contribution})</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="patient-card__details">
          <h4>Clinical Snapshot</h4>
          <div className="patient-card__grid">
            {Object.entries(record).map(([key, value]) => (
              <div className="patient-card__grid-item" key={key}>
                <span className="patient-card__grid-label">{key}</span>
                <span className="patient-card__grid-value">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
        <p className="patient-card__disclaimer">
          Synthetic output for prototyping only. Not a medical diagnostic tool.
        </p>
      </section>
    </div>
  );
};

PatientCard.propTypes = {
  record: PropTypes.object.isRequired,
  prediction: PropTypes.shape({
    probability: PropTypes.number,
    risk_category: PropTypes.string,
    confidence: PropTypes.number,
    explanation: PropTypes.shape({
      top_features: PropTypes.arrayOf(
        PropTypes.shape({
          name: PropTypes.string,
          value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
          contribution: PropTypes.number
        })
      )
    })
  })
};

PatientCard.defaultProps = {
  prediction: {
    probability: 0,
    risk_category: 'Unknown',
    confidence: 0,
    explanation: { top_features: [] }
  }
};

export default PatientCard;
