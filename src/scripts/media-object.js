class Media {
    constructor(unprocessedMediaPath, date, latitudeCoords, longitudeCoords) {
        this._unprocessedMediaPath = unprocessedMediaPath;
        this._processedMediaPath = '';
        this._date = date;
        this._latitudeCoords = latitudeCoords;
        this._longitudeCoords = longitudeCoords;
        this._flagged = false;
    }

    get latitude() {
        return this._latitudeCoords;
    }

    get longitude() {
        return this._longitudeCoords;
    }

    get date() {
        return this._date;
    }

    get unprocessed() {
        return this._unprocessedMediaPath;
    }

    set processed(processedMediaPath) {
        this._processedMediaPath = processedMediaPath;
    }

    get processed() {
        return this._processedMediaPath;
    }

    get flagged() {
        return this._flagged;
    }

    set flagged(value) {
        this._flagged = value;
    }

    toJSON() {
        return {
            unprocessedMediaPath: this._unprocessedMediaPath,
            processedMediaPath: this._processedMediaPath,
            date: this._date,
            latitudeCoords: this._latitudeCoords,
            longitudeCoords: this._longitudeCoords,
            flagged: this._flagged
        };
    }
}

if (typeof window !== 'undefined') {
    window.Media = Media;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = Media;
}
