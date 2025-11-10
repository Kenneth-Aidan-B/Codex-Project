import React from 'react';
import PropTypes from 'prop-types';
import './PatientForm.css';

const numericFields = [
  { name: 'birth_gestation_weeks', label: 'Gestation Weeks', type: 'number', min: 24, max: 42 },
  { name: 'birth_weight_g', label: 'Birth Weight (g)', type: 'number', min: 450, max: 4500 },
  { name: 'apgar_1min', label: 'APGAR 1 min', type: 'number', min: 0, max: 10 },
  { name: 'apgar_5min', label: 'APGAR 5 min', type: 'number', min: 0, max: 10 },
  { name: 'nicu_days', label: 'NICU Days', type: 'number', min: 0, max: 60 },
  { name: 'bilirubin_mg_dL', label: 'Bilirubin (mg/dL)', type: 'number', step: 0.1, min: 0, max: 40 }
];

const binaryFields = [
  { name: 'premature', label: 'Premature (<37w)' },
  { name: 'ventilator', label: 'Ventilator' },
  { name: 'maternal_ototoxic_meds', label: 'Maternal Ototoxic Meds' },
  { name: 'maternal_diabetes', label: 'Maternal Diabetes' },
  { name: 'maternal_hypertension', label: 'Maternal Hypertension' },
  { name: 'alcohol_or_drug_exposure', label: 'Alcohol/Drug Exposure' },
  { name: 'family_history_hearing_loss', label: 'Family History Hearing Loss' },
  { name: 'phototherapy', label: 'Phototherapy' },
  { name: 'exchange_transfusion', label: 'Exchange Transfusion' },
  { name: 'sepsis_or_meningitis', label: 'Sepsis / Meningitis' },
  { name: 'ear_anatomy_abnormality', label: 'Ear Anatomy Abnormality' },
  { name: 'consent_for_research', label: 'Consent for Research' }
];

const selectFields = [
  {
    name: 'sex',
    label: 'Sex',
    options: [
      { value: 'M', label: 'Male' },
      { value: 'F', label: 'Female' }
    ]
  },
  {
    name: 'maternal_infection',
    label: 'Maternal Infection',
    options: [
      'none',
      'CMV',
      'Rubella',
      'Toxoplasmosis',
      'Syphilis',
      'Zika'
    ].map((value) => ({ value, label: value }))
  },
  {
    name: 'genetic_condition',
    label: 'Genetic Condition',
    options: [
      'none',
      'GJB2',
      'Pendred',
      'Usher',
      'Other'
    ].map((value) => ({ value, label: value }))
  },
  {
    name: 'oae_result',
    label: 'OAE Result',
    options: [
      { value: 'pass', label: 'Pass' },
      { value: 'refer', label: 'Refer' }
    ]
  },
  {
    name: 'aabr_result',
    label: 'AABR Result',
    options: [
      { value: 'pass', label: 'Pass' },
      { value: 'refer', label: 'Refer' }
    ]
  }
];

const PatientForm = ({ formData, onChange, onSubmit, submitting }) => (
  <form className="patient-form" onSubmit={onSubmit}>
    <fieldset className="patient-form__fieldset">
      <legend>Demographics</legend>
      <div className="patient-form__grid">
        <label>
          First Name
          <input
            name="first_name"
            type="text"
            value={formData.first_name}
            onChange={onChange}
            required
          />
        </label>
        <label>
          Last Name
          <input
            name="last_name"
            type="text"
            value={formData.last_name}
            onChange={onChange}
            required
          />
        </label>
        {selectFields
          .filter((field) => ['sex', 'maternal_infection', 'genetic_condition'].includes(field.name))
          .map((field) => (
            <label key={field.name}>
              {field.label}
              <select name={field.name} value={formData[field.name]} onChange={onChange}>
                {field.options.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          ))}
      </div>
    </fieldset>

    <fieldset className="patient-form__fieldset">
      <legend>Birth & Clinical Scores</legend>
      <div className="patient-form__grid">
        {numericFields.map((field) => (
          <label key={field.name}>
            {field.label}
            <input
              name={field.name}
              type={field.type}
              value={formData[field.name]}
              onChange={onChange}
              min={field.min}
              max={field.max}
              step={field.step || 1}
              required
            />
          </label>
        ))}
        {selectFields
          .filter((field) => ['oae_result', 'aabr_result'].includes(field.name))
          .map((field) => (
            <label key={field.name}>
              {field.label}
              <select name={field.name} value={formData[field.name]} onChange={onChange}>
                {field.options.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          ))}
      </div>
    </fieldset>

    <fieldset className="patient-form__fieldset">
      <legend>Risk Factors</legend>
      <div className="patient-form__grid patient-form__grid--compact">
        {binaryFields.map((field) => (
          <label key={field.name} className="patient-form__checkbox">
            <input
              type="checkbox"
              name={field.name}
              checked={Boolean(formData[field.name])}
              onChange={(event) =>
                onChange({
                  target: {
                    name: field.name,
                    type: 'checkbox',
                    checked: event.target.checked
                  }
                })
              }
            />
            <span>{field.label}</span>
          </label>
        ))}
      </div>
    </fieldset>

    <button className="patient-form__submit" type="submit" disabled={submitting}>
      {submitting ? 'Submitting...' : 'Add Patient & Predict'}
    </button>
  </form>
);

PatientForm.propTypes = {
  formData: PropTypes.object.isRequired,
  onChange: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
  submitting: PropTypes.bool
};

PatientForm.defaultProps = {
  submitting: false
};

export default PatientForm;
