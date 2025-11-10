import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import './ProgressBar.css';

const ProgressBar = ({ value, label }) => {
  const percentage = Math.round(value * 100);

  const progressClass = classNames('progress-bar__fill', {
    'progress-bar__fill--low': value <= 0.25,
    'progress-bar__fill--moderate': value > 0.25 && value <= 0.6,
    'progress-bar__fill--high': value > 0.6
  });

  return (
    <div className="progress-bar">
      <div className={progressClass} style={{ width: `${percentage}%` }} />
      <span className="progress-bar__label">{label} ({percentage}%)</span>
    </div>
  );
};

ProgressBar.propTypes = {
  value: PropTypes.number.isRequired,
  label: PropTypes.string.isRequired
};

export default ProgressBar;
