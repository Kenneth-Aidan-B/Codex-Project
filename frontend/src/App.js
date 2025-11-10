import React, { useCallback, useMemo, useState } from 'react';
import './App.css';
import PatientCard from './components/PatientCard';
import PatientForm from './components/PatientForm';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const initialFormState = {
  first_name: 'Aurora',
  last_name: 'Starling',
  sex: 'F',
  birth_gestation_weeks: 39,
  premature: 0,
  birth_weight_g: 3200,
  apgar_1min: 8,
  apgar_5min: 9,
  nicu_days: 1,
  ventilator: 0,
  maternal_infection: 'none',
  maternal_ototoxic_meds: 0,
  maternal_diabetes: 0,
  maternal_hypertension: 0,
  alcohol_or_drug_exposure: 0,
  family_history_hearing_loss: 0,
  genetic_condition: 'none',
  bilirubin_mg_dL: 6.2,
  phototherapy: 0,
  exchange_transfusion: 0,
  sepsis_or_meningitis: 0,
  ear_anatomy_abnormality: 0,
  oae_result: 'pass',
  aabr_result: 'pass',
  consent_for_research: 1
};

const App = () => {
  const [searchName, setSearchName] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchStatus, setSearchStatus] = useState('');
  const [formData, setFormData] = useState(initialFormState);
  const [formSubmitting, setFormSubmitting] = useState(false);
  const [formStatus, setFormStatus] = useState('');
  const [formError, setFormError] = useState('');

  const sanitizePayload = useCallback((data) => {
    const payload = { ...data };
    Object.keys(payload).forEach((key) => {
      const value = payload[key];
      if (typeof value === 'string' && value.trim() === '') {
        payload[key] = null;
        return;
      }
      if (['birth_gestation_weeks', 'birth_weight_g', 'apgar_1min', 'apgar_5min', 'nicu_days'].includes(key)) {
        payload[key] = Number(value);
      }
      if (key === 'bilirubin_mg_dL') {
        payload[key] = Number(value);
      }
      if (typeof value === 'boolean') {
        payload[key] = value ? 1 : 0;
      }
    });
    payload.premature = payload.birth_gestation_weeks < 37 ? 1 : Number(payload.premature || 0);
    return payload;
  }, []);

  const handleInputChange = useCallback((event) => {
    const { name, value, type, checked } = event.target;
    setFormData((prev) => {
      let newValue;
      if (type === 'checkbox') {
        newValue = checked ? 1 : 0;
      } else if (type === 'number') {
        newValue = value === '' ? '' : Number(value);
      } else {
        newValue = value;
      }

      const updated = { ...prev, [name]: newValue };
      if (name === 'birth_gestation_weeks') {
        updated.premature = Number(newValue < 37);
      }
      return updated;
    });
  }, []);

  const handleSearch = async () => {
    if (!searchName.trim()) {
      setSearchStatus('Enter a name to search.');
      return;
    }
    setSearchStatus('Searching...');
    setSearchResults([]);
    try {
      const response = await fetch(`${API_BASE_URL}/patient/${encodeURIComponent(searchName.trim())}`);
      if (!response.ok) {
        throw new Error('Patient not found');
      }
      const data = await response.json();
      setSearchResults(data.matches || []);
      setSearchStatus(`Found ${data.matches.length} match(es).`);
      setFormError('');
    } catch (error) {
      setSearchResults([]);
      setSearchStatus('');
      setFormStatus('');
      setFormError(error.message);
    }
  };

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    setFormSubmitting(true);
    setFormStatus('Submitting new patient...');
    setFormError('');
    try {
      const payload = sanitizePayload(formData);
      const response = await fetch(`${API_BASE_URL}/add_patient`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        const errorBody = await response.json();
        throw new Error(errorBody.detail || 'Failed to add patient');
      }
      const data = await response.json();
      setFormStatus('Patient processed successfully.');
      setFormError('');
      setSearchResults([data]);
    } catch (error) {
      setFormError(error.message);
      setFormStatus('');
    } finally {
      setFormSubmitting(false);
    }
  };

  const disclaimer = useMemo(
    () => 'Synthetic model outputs for prototyping only. Not a medical diagnostic tool.',
    []
  );

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">Synthetic Newborn Hearing Risk Explorer</h1>
        <p className="app__subtitle">{disclaimer}</p>
      </header>

      <div className="app__grid">
        <section className="app__section">
          <div className="app__search-box">
            <input
              type="text"
              placeholder="Search patient by name..."
              value={searchName}
              onChange={(event) => setSearchName(event.target.value)}
            />
            <button type="button" onClick={handleSearch}>Search</button>
          </div>
          {searchStatus && <p className="app__status">{searchStatus}</p>}
          {formError && <p className="app__error">{formError}</p>}
          {searchResults.map((item, index) => {
            const record = item.record ?? item;
            const prediction = item.prediction;
            return (
              <PatientCard
                key={index}
                record={record}
                prediction={prediction}
              />
            );
          })}
        </section>

        <section className="app__section">
          <PatientForm
            formData={formData}
            onChange={handleInputChange}
            onSubmit={handleFormSubmit}
            submitting={formSubmitting}
          />
          {formStatus && <p className="app__status">{formStatus}</p>}
          {formError && <p className="app__error">{formError}</p>}
        </section>
      </div>
    </div>
  );
};

export default App;
